import time
from util.strategy_vis import GridAnimation, set_config
from env.graph import operator_info

def do_test():
    op_info = operator_info(
        batch_size=2, 
        in_height=3, in_width=3, 
        inplanes=2, 
        kernel_size=2, 
        outplanes=3,
        stride=1)
    set_config(bar="none",
                log_level="ERROR",
                output_file='test.mp4',
                media_dir="out/media")
    start_time = time.time()
    animation = GridAnimation(op_info=op_info,
                            filename=None,
                            a=0.2,
                            borders=3.5,
                            draft=True)
    animation.render()
    end_time = time.time()
    render_time = end_time - start_time
    print(f"Rendered game animation in {render_time:.2f} seconds")