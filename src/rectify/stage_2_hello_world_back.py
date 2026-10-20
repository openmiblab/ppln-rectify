import os
import logging
import pydmr

from template.utils import pipe, demo

PIPELINE = 'template'

def run(build):

    logging.info("Stage 2 --- Hello World Back ---")
    dir_output = pipe.stage_output_dir(build, PIPELINE, __file__)

    text_file = os.path.join(dir_output, 'hello_world.txt')
    demo.append_string_to_file('Hello World Back', text_file)

    csv_file = os.path.join(dir_output, 'hello_world.csv')
    demo.append_string_to_csv('Hello World Back', csv_file)

    dmr_file = os.path.join(dir_output, 'hello_world_back.dmr')
    write_to_dmr(dmr_file)

    logging.info("Stage 2. Finished Hello World Back.")



def write_to_dmr(file_path: str):
    """Writes a string to a dmr file."""
    dmr = {
        'data':{
            'call': ['Callout', '', 'str'],
            'response': ['Response to callout', '', 'str'],
        },
        'pars':{
            ('me', 'callout', 'call'): 'Hello World',
            ('you', 'callout', 'response'): 'Hello World Back'
        }
    }
    pydmr.write(file_path, dmr)

if __name__ == '__main__':

    BUILD = r"C:\Users\md1spsx\Documents\Data\template"
    pipe.run_script(run, BUILD, PIPELINE)