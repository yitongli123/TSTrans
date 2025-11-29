import math

from lib.models.tstrans import build_tstrans
from lib.test.tracker.basetracker import BaseTracker
import torch

from lib.test.tracker.vis_utils import gen_visualization
from lib.test.utils.hann import hann2d
from lib.train.data.processing_utils import sample_target
# for debug
import cv2
import os

from lib.test.tracker.data_utils import Preprocessor
from lib.utils.box_ops import clip_box
from lib.utils.ce_utils import generate_mask_cond


class TSTrans(BaseTracker):
    def __init__(self, params, dataset_name):
        super(TSTrans, self).__init__(params)
        network = build_tstrans(params.cfg, training=False)
        network.load_state_dict(torch.load(self.params.checkpoint, map_location='cpu')['net'], strict=True)
        self.cfg = params.cfg
        self.network = network.cuda()  # cuda()
        self.network.eval()
        self.preprocessor = Preprocessor()
        self.state = None

        self.feat_sz = self.cfg.TEST.SEARCH_SIZE // self.cfg.MODEL.BACKBONE.STRIDE
        # motion constrain
        self.output_window = hann2d(torch.tensor([self.feat_sz, self.feat_sz]).long(), centered=True).cuda()  # cuda()

        # for debug
        self.debug = params.debug
        self.use_visdom = params.debug
        self.frame_id = 0
        if self.debug:
            if not self.use_visdom:
                self.save_dir = "debug"
                if not os.path.exists(self.save_dir):
                    os.makedirs(self.save_dir)
            else:
                # self.add_hook()
                self._init_visdom(None, 1)
        # for save boxes from all queries
        self.save_all_boxes = params.save_all_boxes
        self.num_temporal_templates = self.cfg.DATA.TEMPLATE.L
        self.state_list = []
        self.motion_cache = []
        self.distance_cache = []
        self.temporal_t_ids = []
        for i in range(self.num_temporal_templates):
            self.temporal_t_ids.append(0)

    def initialize(self, images, info: dict):
        # save states
        self.frames = images
        self.state = info['init_bbox']
        self.w = info['init_bbox'][2]
        self.h = info['init_bbox'][3]
        self.frame_id = 0
        self.state_list.append(self.state)
        # print('save_all_boxes:', self.save_all_boxes) -> False
        if self.save_all_boxes:      
            '''save all predicted boxes'''
            all_boxes_save = info['init_bbox'] * self.cfg.MODEL.NUM_OBJECT_QUERIES
            return {"all_boxes": all_boxes_save}

    def frame_template_initialize(self, template_list, bbox_list):
        z_dict_list = []
        box_mask_z_list = []
        for i in range((self.num_temporal_templates+1)):
            z_patch_arr, resize_factor, z_amask_arr = sample_target(template_list[i], bbox_list[i], self.params.template_factor,
                                                        output_sz=self.params.template_size)
            template = self.preprocessor.process(z_patch_arr, z_amask_arr)

            if self.cfg.MODEL.BACKBONE.CE_LOC:
                template_bbox = self.transform_bbox_to_crop(bbox_list[i], resize_factor,
                                                            template.tensors.device).squeeze(1)
                box_mask_z = generate_mask_cond(self.cfg, 1, template.tensors.device, template_bbox)
            
            z_dict_list.append(template.tensors.unsqueeze(0))
            box_mask_z_list.append(box_mask_z)
            
        with torch.no_grad():
            z_dict = torch.cat(z_dict_list, dim=0)
            if box_mask_z_list[0] is None:
                box_mask_z = None
            else:
                box_mask_z = torch.cat(box_mask_z_list, dim=1)
    
        return z_dict, box_mask_z

    def pred_motion_fusing(self, prediction, motion_history, prediction_last):
        N = len(motion_history)
        x_move = 0.0
        y_move = 0.0
        if N != 1:
            x_move = (motion_history[N-1][0] - motion_history[0][0]) / (N-1)
            y_move = (motion_history[N-1][1] - motion_history[0][1]) / (N-1)
        self.distance_cache.append([x_move, y_move])
        if N == 1 or N == 2:
            fusing_result = prediction
        else:
            x_move_mean = 0.0
            y_move_mean = 0.0
            for j in range(N-1):
                x_move_mean += self.distance_cache[j+1][0]
                y_move_mean += self.distance_cache[j+1][1]
            x_move_mean = x_move_mean / (N-1)
            y_move_mean = y_move_mean / (N-1)
            x_motion = prediction_last[0] + x_move_mean
            y_motion = prediction_last[1] + y_move_mean

            if math.sqrt((prediction[0]-x_motion)**2+(prediction[1]-y_motion)**2) < self.cfg.MODEL.MOTION.GAMA:  
                fusing_result = prediction
            else:
                fusing_result = [x_motion, y_motion, prediction[2], prediction[3]]
            
        return fusing_result

    def track(self, image, info: dict = None):
        H, W, _ = image.shape
        self.frame_id += 1
        x_patch_arr, resize_factor, x_amask_arr = sample_target(image, self.state, self.params.search_factor,
                                                                output_sz=self.params.search_size)  # (x1, y1, w, h)
        search = self.preprocessor.process(x_patch_arr, x_amask_arr)

        with torch.no_grad():
            bbox_list = [self.state_list[0]]
            template_list = [self.frames[0]]
            x_dict = search
            #print(self.frame_id, ':', self.temporal_t_ids)
            for j in range(self.num_temporal_templates):
                template_list.append(self.frames[self.temporal_t_ids[j]])
                bbox_list.append(self.state_list[self.temporal_t_ids[j]])
            
            z_dict, box_mask_z = self.frame_template_initialize(template_list, bbox_list)
            
            out_dict = self.network.forward(
                template=z_dict, search=x_dict.tensors, pattern='tracking', num_temporal_templates=self.num_temporal_templates, ce_template_mask=box_mask_z)
            #print('ce_template_mask:', box_mask_z.shape) -> (1, 256), 256=(1+3)*(128/16)*(128/16)
            #print('search:', x_dict.tensors.shape) -> (1, 3, 256, 256)
            #print('template:', z_dict.shape) -> (4, 1, 3, 256, 256)
        # add hann windows
        pred_score_map = out_dict['score_map']
        response = self.output_window * pred_score_map
        pred_centers = self.network.box_head.cal_bbox(response, out_dict['offset_map'])
        pred_centers = pred_centers.view(-1, 2)
        # Baseline: Take the mean of all pred boxes as the final result
        pred_center = (pred_centers.mean(
            dim=0) * self.params.search_size / resize_factor).tolist()  # (cx, cy) [0,1]
        pred_box = pred_center
        pred_box.append(self.w)
        pred_box.append(self.h)
        pred_box = self.map_box_back(pred_box, resize_factor)  # pred_box: (cx, cy, w, h) -> (x1, y1, w, h)
        
        # motion correction        
        if len(self.motion_cache) < self.cfg.MODEL.MOTION.BETA:
            self.motion_cache.append(pred_box)  # (x1, y1, w, h)
        else:
            self.motion_cache = self.motion_cache[1:]          
            self.motion_cache.append(pred_box)  # (x1, y1, w, h)
        self.state = self.pred_motion_fusing(pred_box, self.motion_cache, self.state_list[-1])  # (x1, y1, w, h)

        # get the final box result
        self.state = clip_box(self.state, H, W, margin=0)  # type: list
        self.state_list.append(self.state)

        # update the temporal templates
        max_score, _ = torch.max(response.flatten(1), dim=1, keepdim=True)
        if max_score[0][0] >= self.cfg.DATA.TEMPLATE.CONFIDENCE and self.frame_id % 10 == 0:  # self.frame_id starts from 1 (the 2rd frame)
            for q in range((self.num_temporal_templates-1)):
                self.temporal_t_ids[q] = self.temporal_t_ids[(q+1)]
            self.temporal_t_ids[(self.num_temporal_templates-1)] = self.frame_id

        # for debug
        if self.debug:
            if not self.use_visdom:
                x1, y1, w, h = self.state
                image_BGR = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                cv2.rectangle(image_BGR, (int(x1),int(y1)), (int(x1+w),int(y1+h)), color=(0,0,255), thickness=2)
                save_path = os.path.join(self.save_dir, "%04d.jpg" % self.frame_id)
                cv2.imwrite(save_path, image_BGR)
            else:
                self.visdom.register(self.frame_id, 'text', 1, 'frame')
                self.visdom.register((image, info['gt_bbox'].tolist(), self.state), 'Tracking', 1, 'Tracking')

                self.visdom.register(torch.from_numpy(x_patch_arr).permute(2, 0, 1), 'image', 1, 'search_region')
                
                z_patch_arr, resize_factor, z_amask_arr = sample_target(template_list[0], bbox_list[0], self.params.template_factor,
                                                    output_sz=self.params.template_size)
                self.visdom.register(torch.from_numpy(z_patch_arr).permute(2, 0, 1), 'image', 1, 'template_standard')
                for k in range(self.num_temporal_templates):
                    z_patch_arr, resize_factor, z_amask_arr = sample_target(template_list[k+1], bbox_list[k+1], self.params.template_factor,
                                                        output_sz=self.params.template_size)
                    self.visdom.register(torch.from_numpy(z_patch_arr).permute(2, 0, 1), 'image', 1, 'template_temporal_'+str(k+1))
                
                self.visdom.register(pred_score_map.view(self.feat_sz, self.feat_sz), 'heatmap', 1, 'score_map')
                self.visdom.register((pred_score_map * self.output_window).view(self.feat_sz, self.feat_sz), 'heatmap', 1, 'score_map_hann')

                if 'removed_indexes_s' in out_dict and out_dict['removed_indexes_s']:
                    removed_indexes_s = out_dict['removed_indexes_s']
                    removed_indexes_s = [removed_indexes_s_i.cpu().numpy() for removed_indexes_s_i in removed_indexes_s]
                    masked_search = gen_visualization(x_patch_arr, removed_indexes_s)
                    self.visdom.register(torch.from_numpy(masked_search).permute(2, 0, 1), 'image', 1, 'masked_search')

                while self.pause_mode:
                    if self.step:
                        self.step = False
                        break

        return {"target_bbox": self.state}

    def map_box_back(self, pred_box: list, resize_factor: float):
        cx_prev, cy_prev = self.state[0] + 0.5 * self.state[2], self.state[1] + 0.5 * self.state[3]
        cx, cy, w, h = pred_box
        half_side = 0.5 * self.params.search_size / resize_factor
        cx_real = cx + (cx_prev - half_side)
        cy_real = cy + (cy_prev - half_side)
        return [cx_real - 0.5 * w, cy_real - 0.5 * h, w, h]

    def map_box_back_batch(self, pred_box: torch.Tensor, resize_factor: float):
        cx_prev, cy_prev = self.state[0] + 0.5 * self.state[2], self.state[1] + 0.5 * self.state[3]
        cx, cy, w, h = pred_box.unbind(-1) # (N,4) --> (N,)
        half_side = 0.5 * self.params.search_size / resize_factor
        cx_real = cx + (cx_prev - half_side)
        cy_real = cy + (cy_prev - half_side)
        return torch.stack([cx_real - 0.5 * w, cy_real - 0.5 * h, w, h], dim=-1)

    def add_hook(self):
        conv_features, enc_attn_weights, dec_attn_weights = [], [], []

        for i in range(12):
            self.network.backbone.blocks[i].attn.register_forward_hook(
                # lambda self, input, output: enc_attn_weights.append(output[1])
                lambda self, input, output: enc_attn_weights.append(output[1])
            )

        self.enc_attn_weights = enc_attn_weights


def get_tracker_class():
    return TSTrans
