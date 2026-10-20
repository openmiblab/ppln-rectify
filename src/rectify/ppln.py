import template as ppln
from template.utils import pipe

PIPELINE = 'template'

def run(build):
    
    ppln.stage_1_hello_world.run(build)
    ppln.stage_2_hello_world_back.run(build)


if __name__=='__main__':

    BUILD = r"C:\Users\md1spsx\Documents\Data\template"
    pipe.run_script(run, BUILD, PIPELINE)