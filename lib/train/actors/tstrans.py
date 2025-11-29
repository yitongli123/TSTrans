from . import BaseActor
from lib.utils.misc import NestedTensor
from lib.utils.box_ops import box_cxcywh_to_xyxy, box_xywh_to_xyxy, box_xywh_to_cxcywh
import torch
from lib.utils.merge import merge_template_search
from ...utils.heapmap_utils import generate_heatmap
from ...utils.ce_utils import generate_mask_cond, adjust_keep_rate


class TSTransActor(BaseActor):
    """ Actor for training TSTrans models """

    def __init__(self, net, objective, loss_weight, settings, cfg=None):
        super().__init__(net, objective)
        self.loss_weight = loss_weight
        self.settings = settings
        self.bs = self.settings.batchsize  # batch size
        self.cfg = cfg

    def __call__(self, data):
        """
        args:
            data - The input data, should contain the fields 'template', 'search', 'gt_bbox'.
            template_images: (N_t, batch, 3, H, W)
            search_images: (N_s, batch, 3, H, W)
        returns:
            loss    - the training loss
            status  -  dict containing detailed losses
        """
        # forward pass
        out_dict = self.forward_pass(data)

        # compute losses
        loss, status = self.compute_losses(out_dict, data)

        return loss, status

    def forward_pass(self, data):
        # checking
        assert len(data['template_images']) == 2  # 1 for standard template, 1 for temporal templates
        assert len(data['search_images']) == self.cfg.DATA.SEARCH.NUMBER

        template_list = data['template_images']
        # template_list: Tensor, torch.Size([2, batch, 3, 128, 128])

        search_img = data['search_images'][0].view(-1, *data['search_images'].shape[2:])

        box_mask_z = None
        ce_keep_rate = None
        if self.cfg.MODEL.BACKBONE.CE_LOC:  
            box_mask_z_list = []   
            for i in range(len(data['template_images'])): 
                box_mask_z_list.append(generate_mask_cond(self.cfg, template_list[i].shape[0], template_list[i].device,
                                            data['template_anno'][i]))
            box_mask_z = torch.cat(box_mask_z_list, dim=1)
            
            ce_start_epoch = self.cfg.TRAIN.CE_START_EPOCH
            ce_warm_epoch = self.cfg.TRAIN.CE_WARM_EPOCH
            ce_keep_rate = adjust_keep_rate(data['epoch'], warmup_epochs=ce_start_epoch,
                                                total_epochs=ce_start_epoch + ce_warm_epoch,
                                                ITERS_PER_EPOCH=1,
                                                base_keep_rate=self.cfg.MODEL.BACKBONE.CE_KEEP_RATIO[0])

        out_dict = self.net(template=template_list,
                            search=search_img,
                            pattern='learning',
                            num_temporal_templates=self.cfg.DATA.TEMPLATE.L,
                            ce_template_mask=box_mask_z,
                            ce_keep_rate=ce_keep_rate,                          
                            return_last_attn=False)
       
        return out_dict

    def compute_losses(self, pred_dict, gt_dict, return_status=True): 
        # gt gaussian map
        gt_bbox = gt_dict['search_anno'][-1]  # (Ns, batch, 4) (x1,y1,w,h) -> (batch, 4)
        gt_gaussian_maps = generate_heatmap(gt_dict['search_anno'], self.cfg.DATA.SEARCH.SIZE, self.cfg.MODEL.BACKBONE.STRIDE)
        gt_gaussian_maps = gt_gaussian_maps[-1].unsqueeze(1)

        pred_centers = pred_dict['pred_center']  # shape: torch.Size([batch, 1, 2])
        if torch.isnan(pred_centers).any():
            raise ValueError("Network outputs is NAN! Stop Training")
        num_queries = pred_centers.size(1)
        assert num_queries == 1
        pred_centers = pred_centers.view(-1, 2)
        gt_centers = box_xywh_to_cxcywh(gt_bbox)[:, :2].clamp(min=0.0, max=1.0)  # (batch,2)

        # compute l1 loss
        l1_loss = self.objective['l1'](pred_centers, gt_centers)
        # compute location loss
        if 'score_map' in pred_dict:
            location_loss = self.objective['focal'](pred_dict['score_map'], gt_gaussian_maps)
        else:
            location_loss = torch.tensor(0.0, device=l1_loss.device)
        # weighted sum
        loss = self.loss_weight['l1'] * l1_loss + self.loss_weight['focal'] * location_loss
        if return_status:
            # status for log
            status = {"Loss/total": loss.item(),
                      "Loss/l1": l1_loss.item(),
                      "Loss/location": location_loss.item(),
                      }
            return loss, status
        else:
            return loss
