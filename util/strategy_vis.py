import os
from manim import (config,
                   Scene, VGroup, Square, Line,
                   ApplyMethod, Create, FadeOut, ScaleInPlace,
                   BLUE, PURPLE, GRAY, GREEN, RED, GREY_B, ORANGE, YELLOW_E, interpolate_color,
                   LEFT, RIGHT, UP, DOWN)
from env.graph import operator_info

# class operator_info:
#     def __init__(self, batch_size:int=1, in_height:int=16, in_width:int = 16 , inplanes:int = 3, 
#                  outplanes:int = 1, kernel_size:int = 3, stride:int = 1) -> None:
#         self.batch_size = batch_size
#         self.in_height = in_height
#         self.in_width = in_width
#         self.inplanes = inplanes
#         self.outplanes = outplanes
#         self.kernel_size = kernel_size
#         self.stride = stride
    
#     def __str__(self) -> str:
#         return "batch_size: %d, in_height: %d, in_width: %d, inplanes: %d, outplanes: %d, kernel_size: %d, stride: %d" % \
#                 (self.batch_size, self.in_height, self.in_width, self.inplanes, self.outplanes, self.kernel_size, self.stride)
    
#     def __repr__(self) -> str:
#         return self.__str__()

def set_config(bar:str='display', 
               log_level:str="ERROR", 
               frame_rate:int=30,
               quality:str="720p",
               output_file:str="tmp.mp4", 
               media_dir:str="out/media"):
    
    # config.progress_bar = bar
    config.frame_rate = frame_rate
    # config.frame_size = (1280, 720) if quality == "720p" else (1920, 1080)
    config.ffmpeg_loglevel = log_level
    # config.verbosity = log_level
    config.output_file = output_file
    config.media_dir = media_dir
    
class GridAnimation(Scene):
    def __init__(self,
                 op_info:operator_info,
                 filename:str,
                 a:float=0.15,
                 borders:float=0.5,
                 detail:bool=False,
                 draft:bool=False,
                 **kwargs):
        super().__init__(**kwargs)
        self.batch_size = op_info.batch_size
        self.in_height = op_info.in_height
        self.in_width = op_info.in_width
        self.in_channel = op_info.inplanes
        self.out_channel = op_info.outplanes
        self.kernel_size = op_info.kernel_size
        self.stride = op_info.stride
        self.out_width = 1 + (self.in_width - self.kernel_size)//self.stride
        self.out_height = 1 + (self.in_height - self.kernel_size)//self.stride
        self.dot_width = self.kernel_size * self.out_width
        self.dot_height = self.kernel_size * self.out_height
        self.dot_channel = self.in_channel * self.out_channel
        self.img_num = self.batch_size * self.in_height * self.in_width * self.in_channel
        self.kern_num = self.out_channel * self.kernel_size * self.kernel_size * self.in_channel
        self.input_num = self.img_num + self.kern_num
        self.output_num = self.batch_size * self.out_channel * self.out_width * self.out_height
        self.filename = filename
        # detail: whether to show img and kern
        self.detail = detail
        # draft: only for test render create/fadeout
        self.draft = draft
        
        self.a = a # related to pixel size
        self.borders = borders # size of borders
        self.gen_scale = 0.25 * (self.a / 0.15)
        self.dot_scale = self.gen_scale
        self.gen_opacity = 0.8
        self.dot_opacity = 0.99
        self.img_color = BLUE
        self.kern_color = PURPLE
        self.out_color = GREY_B
        self.dot_color = GREY_B
        self.lock_color = GRAY
        self.tmp_color = ORANGE
        self.out_fin_color = GREEN
        
    # def color_map(self, value):
    #     start_color = GREY_B
    #     end_color = ORANGE
    #     interpolated_color = interpolate_color(start_color, end_color, value)
    #     return interpolated_color
    
    def get_lines(self, vgrp_obj, width, height, kernel_size, stroke_width=3.0, color=YELLOW_E):
        """
        draw lines on a plane
        """
        lu = 0
        ru = width-1
        ld = (height-1)*width
        rd = height*width-1
        ## outlines
        lines = [Line(vgrp_obj[lu].get_corner(UP+LEFT), 
                      vgrp_obj[ru].get_corner(UP+RIGHT), 
                      color=color,
                      stroke_width = stroke_width),
                 Line(vgrp_obj[lu].get_corner(UP+LEFT), 
                      vgrp_obj[ld].get_corner(DOWN+LEFT), 
                      color=color,
                      stroke_width = stroke_width),
                 Line(vgrp_obj[ru].get_corner(UP+RIGHT), 
                      vgrp_obj[rd].get_corner(DOWN+RIGHT),
                      color=color,
                      stroke_width=stroke_width),
                 Line(vgrp_obj[ld].get_corner(DOWN+LEFT), 
                      vgrp_obj[rd].get_corner(DOWN+RIGHT),
                      color=color,
                      stroke_width=stroke_width)]
        
        # vertical lines 
        for i in range(width//kernel_size - 1):
            lines.append(Line(vgrp_obj[(i+1)*kernel_size-1].get_corner(UP+RIGHT), 
                              vgrp_obj[ld+(i+1)*kernel_size-1].get_corner(DOWN+RIGHT), 
                              color=color,
                              stroke_width = stroke_width))
        # horizontal lines
        for i in range(height//kernel_size - 1):
            lines.append(Line(vgrp_obj[((i+1)*kernel_size-1)*width].get_corner(LEFT+DOWN), 
                              vgrp_obj[((i+1)*kernel_size-1)*width+ru].get_corner(RIGHT+DOWN), 
                              color=color,
                              stroke_width = stroke_width))
        vgrp_obj.add(*lines)
        return vgrp_obj
    
    def get_grid(self, batch_size, width, height, channel, color=BLUE, 
                 interv=0.4, opacity=0.8, a=0.15,
                 group:int=0, sep_size:int=0, _3d=True):
        """
        Create a width * height grid of squares,
        which has channel layers, and batch_size batchs.
        
        group: determine whether to make channel dim grouped, used for mid node
        """
        plane = VGroup(*[Square(side_length=1.0, 
                                stroke_width=1.5,
                                fill_color=color, 
                                fill_opacity=1).shift((i % width - (width//2)) * RIGHT + ((height//2) - i // height) * UP)
                     for i in range(width*height)]).scale(a)
        if sep_size != 0:
            # for build mid plane add line
            plane = self.get_lines(plane, width, height, sep_size)

        planes_interv = interv * interv
        
        if _3d:
            shifts = [planes_interv*(i + (0 if group == 0 else i // group))*(LEFT+DOWN) for i in range(channel)]
        else:
            shifts = [(planes_interv*(0 if group == 0 else i // group)+(planes_interv + height*a)*i)*(DOWN) for i in range(channel)]
                        
        vol = VGroup(*[plane.copy().shift(shifts[i]).set_opacity(1.0*(opacity**(channel-1-i)))
                       for i in range(channel)])
        if _3d:
            vol_interv = (channel + (0 if group==0 else channel//group))*planes_interv + width*interv/2.0
        else:
            vol_interv = width*a+planes_interv
            
        batch = VGroup(*[vol.copy().shift(vol_interv*i*RIGHT)
                         for i in range(batch_size)])
        return batch
    
    def get_index(self, width, height, channel, node_id, kern_size=0):
        """
        transform relative index of obj to (vol_id, plane_id, id)
        
        kern_size is for dot plane
        """
        plane = width * height
        vol = plane * channel
        vol_id = node_id // vol
        _plane_id = (node_id - vol_id * vol) // plane
        plane_id = channel - 1 - _plane_id # when constrcut plane, the first plane is the last one
        id = node_id - vol_id * vol - _plane_id * plane
        if kern_size != 0:
            # id = (node_id - vol_id * vol - _plane_id * plane) // plane
            block_id = id // (kern_size*kern_size)
            in_block_id = id % (kern_size*kern_size)
            block_num_per_row = width // kern_size
            row_num = (block_id // block_num_per_row) * kern_size + in_block_id // kern_size # rows before node
            col_id = (block_id % block_num_per_row) * kern_size + in_block_id % kern_size    # cols before node
            id = row_num * width + col_id # id start from
        else:
            id = id % plane
        return vol_id, plane_id, id
    
    def move_plane(self, obj:VGroup, _plane_id:int, ht:float, dir, scale=0.4, group:int=0):
        """
        group: determine whether to make channel dim grouped, used for mid node
        add more distance
        """
        initial_position = obj.get_center()
        dist = (_plane_id+1 + (0 if group == 0 else _plane_id // group))*scale*scale+ht
        target_position = initial_position + dist * dir
        move_down_animation = ApplyMethod(obj.move_to, target_position)
        self.play(move_down_animation, run_time=0.25)

    def _compute(self, 
                 obj, obj_wd:int, obj_ht:int, obj_c:int, obj_at_top:bool, 
                 node_id:int, scale:float=0.4, 
                 group:int=0, 
                 kern_size:int=0,
                 _3d:bool=False):
        """
        index: relative to obj
        obj: self.img or self.kern or self.dot or self.out
        
        group: in_channel
        kern_size: kernel size
        """
        dir = DOWN if obj_at_top else UP
        rdir = UP if obj_at_top else DOWN
        
        vol_id, plane_id, id = self.get_index(obj_wd, obj_ht, obj_c, node_id, kern_size)
        if _3d:
            # move in
            self.move_plane(obj[vol_id][plane_id], plane_id, scale*obj_ht, dir, scale, group)
            # to front
            obj[vol_id][plane_id][id].set_z_index(1)
        # zoom in
        scale_animation = ScaleInPlace(obj[vol_id][plane_id][id], scale_factor=1.25)
        self.play(scale_animation, run_time=0.1)
        # set color
        self.play(obj[vol_id][plane_id][id].animate.set_fill(RED), run_time=0.1)
        self.wait(0.1)
        # zoom out
        scale_animation = ScaleInPlace(obj[vol_id][plane_id][id], scale_factor=0.8)
        self.play(scale_animation, run_time=0.1)
        if _3d:
            # to back
            obj[vol_id][plane_id][id].set_z_index(0)
            # move out
            self.move_plane(obj[vol_id][plane_id], plane_id, scale*obj_ht, rdir, scale, group)
        self.wait(0.1)

    def _delete(self, 
                obj, obj_wd:int, obj_ht:int, obj_c:int, obj_co, obj_at_top:bool, 
                node_id:int, scale:float=0.4, 
                group:int=0,
                kern_size:int=0,
                _3d:bool=False):
        dir = DOWN if obj_at_top else UP
        rdir = UP if obj_at_top else DOWN
        vol_id, plane_id, id = self.get_index(obj_wd, obj_ht, obj_c, node_id, kern_size)
        if _3d:
            # move in
            self.move_plane(obj[vol_id][plane_id], plane_id, scale*obj_ht, dir, scale, group)
            # to front
            obj[vol_id][plane_id][id].set_z_index(1)
        # zoom in
        scale_animation = ScaleInPlace(obj[vol_id][plane_id][id], scale_factor=1.25)
        self.play(scale_animation, run_time=0.1)
        # set color
        self.play(obj[vol_id][plane_id][id].animate.set_fill(obj_co), run_time=0.1)
        self.wait(0.1)
        # zoom out
        scale_animation = ScaleInPlace(obj[vol_id][plane_id][id], scale_factor=0.8)
        self.play(scale_animation, run_time=0.1)
        if _3d:
            # to back
            obj[vol_id][plane_id][id].set_z_index(0)
            # move out
            self.move_plane(obj[vol_id][plane_id], plane_id, scale*obj_ht, rdir, scale, group)
        self.wait(0.1)
    
    def _load(self, obj, obj_wd:int, obj_ht:int, obj_c:int, obj_at_top:bool, node_id:int, scale:float=0.4, 
              group:int=0, kern_size:int=0, _3d:bool=False):
        self._compute(obj, obj_wd, obj_ht, obj_c, obj_at_top, node_id, scale, group, kern_size, _3d)
        
    def _store(self, obj, obj_wd:int, obj_ht:int, obj_c:int, obj_co, obj_at_top:bool, node_id:int, scale:float=0.4, 
               group:int=0, kern_size:int=0, _3d:bool=False):
        self._delete(obj, obj_wd, obj_ht, obj_c, obj_co, obj_at_top, node_id, scale, group, kern_size, _3d)

    def do_action(self, action:str, node_id:int, auto:bool=False, _3d:bool=False):
        """
        do action on node_id
        """
        is_input = node_id < self.input_num
        is_output = node_id >= self.input_num and node_id < self.input_num + self.output_num
        # choose animation according to action
        is_dot = (not is_input) and (not is_output)
        is_img = node_id < self.img_num if is_input else False
        # do action on input, kernel and output
        scale = self.dot_scale if is_dot else self.gen_scale
        if action == 'LOAD':
            if is_input and _3d:
                if is_img:
                    self._load(self.img, self.in_width, self.in_height, self.in_channel, True, node_id, scale)
                else:
                    self._load(self.kern, self.kernel_size, self.kernel_size, self.in_channel, True, node_id-self.img_num, scale)
            elif is_output:
                self._load(self.out, self.out_width, self.out_height, self.out_channel, False, node_id-self.input_num, scale)
            else:
                # load on dot(mid) node
                self._load(self.dot, self.dot_width, self.dot_height, self.dot_channel, False, node_id-self.input_num-self.output_num, scale, group=self.in_channel, kern_size=self.kernel_size)
        elif action == 'STORE':
            assert not is_input and "never store input"
            if is_output:
                self._store(self.out, self.out_width, self.out_height, self.out_channel, self.out_fin_color if auto else self.tmp_color, False, node_id-self.input_num, scale)
            else:
                # store on dot(mid) node
                self._store(self.dot, self.dot_width, self.dot_height, self.dot_channel, self.lock_color if auto else self.tmp_color, False, node_id-self.input_num-self.output_num, scale, group=self.in_channel, kern_size=self.kernel_size)
        elif action == 'COMPUTE':
            assert not is_input and "never compute input"
            if is_output:
                self._compute(self.out, self.out_width, self.out_height, self.out_channel, False, node_id-self.input_num, scale)
            else:
                # compute on dot(mid) node
                self._compute(self.dot, self.dot_width, self.dot_height, self.dot_channel, False, node_id-self.input_num-self.output_num, scale, group=self.in_channel, kern_size=self.kernel_size)
        elif action == 'DELETE':
            assert not is_output and "never delete output"
            if is_input and _3d:
                if is_img:
                    self._delete(self.img, self.in_width, self.in_height, self.in_channel, self.lock_color if auto else self.tmp_color, True, node_id, scale)
                else:
                    self._delete(self.kern, self.kernel_size, self.kernel_size, self.in_channel, self.lock_color if auto else self.tmp_color, True, node_id-self.img_num, scale)
            else:
                # delete on dot(mid) node
                assert auto == True and "never delete dot(mid) node actively"
                self._delete(self.dot, self.dot_width, self.dot_height, self.dot_channel, self.lock_color, False, node_id-self.input_num-self.output_num, scale, group=self.in_channel, kern_size=self.kernel_size)
                
    def draw_by_file(self, filename:str, _3d:bool=False):
        """
        draw the operate animation of file
        """
        if os.path.exists(filename):
            with open(filename, "r") as file:
                for line in file.readlines():
                    cmd = line.strip('\n').split(' ')
                    # cmd[2] indicates whether it do autoly
                    self.do_action(cmd[0], int(cmd[1]), cmd[2] == '*', _3d=_3d)
    
    def construct(self):
        render_3d = False # render dot/out vol in 3d or not
        dot_pos = LEFT + UP
        out_pos = RIGHT + UP
        
        if self.detail:
            self.img = self.get_grid(batch_size=self.batch_size,
                                     width=self.in_width, 
                                     height=self.in_height,
                                     channel=self.in_channel,
                                     color=self.img_color,
                                     interv=self.gen_scale,
                                     a=self.a,
                                     opacity=self.gen_opacity,
                                     _3d=True).to_edge(LEFT+UP, buff=self.borders)

            self.kern = self.get_grid(batch_size=self.out_channel,
                                      width=self.kernel_size,
                                      height=self.kernel_size,
                                      channel=self.in_channel,
                                      color=self.kern_color,
                                      interv=self.gen_scale,
                                      a=self.a,
                                      opacity=self.gen_opacity,
                                      _3d=True).to_edge(RIGHT+UP, buff=self.borders)
            render_3d = True
            dot_pos = LEFT+DOWN
            out_pos = RIGHT+DOWN
            
        self.dot = self.get_grid(batch_size=self.batch_size,
                                    width=self.dot_width,
                                    height=self.dot_height,
                                    channel=self.dot_channel,
                                    color=self.dot_color,
                                    interv=self.dot_scale,
                                    opacity=self.dot_opacity,
                                    a=self.a,
                                    group=self.in_channel,
                                    sep_size=self.kernel_size,
                                    _3d=render_3d).to_edge(dot_pos, buff=self.borders)

        self.out = self.get_grid(batch_size=self.batch_size,
                                    width=self.out_width,
                                    height=self.out_height,
                                    channel=self.out_channel,
                                    color=self.out_color,
                                    interv=self.gen_scale,
                                    a=self.a,
                                    opacity=self.gen_opacity,
                                    _3d=render_3d).to_edge(out_pos, buff=self.borders)
        if not self.detail:
            self.dot.to_edge(UP, buff=0.5)
            self.out.to_edge(UP, buff=0.5)
        grids = []
        if self.detail:
            grids.append(self.img)
            grids.append(self.kern)
        grids.append(self.dot)
        grids.append(self.out)

        creates = [Create(grid, lag_ratio=0) for grid in grids]
        fade_outs = [FadeOut(grid) for grid in grids]
        
        self.play(*creates) # create grids
        
        if self.draft:
            self.wait(0.5)
        else:
            self.draw_by_file(self.filename)
        
        self.play(*fade_outs, run_time=0.25) # exit