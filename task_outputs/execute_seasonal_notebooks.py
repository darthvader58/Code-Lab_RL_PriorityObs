"""Execute only the new seasonal notebook sections, preserving historical outputs and exports.

Uses an in-process IPython shell to execute the actual cell sources and capture rich MIME outputs.
No extraction/CZML export cells run. The execution scope is recorded in each notebook's metadata.
"""
import sys
from pathlib import Path

import nbformat
from IPython.core.interactiveshell import InteractiveShell
from IPython.utils.capture import capture_output

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from baselines.notebook_seasonal import attach, TAG


def main():
    for mission,path in [('earthcare','missions/earthcare/EarthCare_trackwise.ipynb'),
                         ('trmm','missions/trmm/TRMM_trackwise.ipynb'),
                         ('comparison','missions/comparison/E12_comparison.ipynb')]:
        target=ROOT/path
        nb=attach(nbformat.read(target,as_version=4),mission)
        InteractiveShell.clear_instance()
        shell=InteractiveShell.instance(user_ns={})
        count=0
        for cell in nb.cells:
            if TAG not in cell.metadata.get('tags',[]) or cell.cell_type!='code': continue
            count+=1
            with capture_output() as captured:
                result=shell.run_cell(cell.source,store_history=False)
            if result.error_before_exec or result.error_in_exec:
                raise RuntimeError(f'{mission} seasonal cell {count} failed: {captured.stdout} {captured.stderr}') from (result.error_before_exec or result.error_in_exec)
            outputs=[]
            if captured.stdout: outputs.append(nbformat.v4.new_output('stream',name='stdout',text=captured.stdout))
            if captured.stderr: outputs.append(nbformat.v4.new_output('stream',name='stderr',text=captured.stderr))
            for rich in captured.outputs:
                outputs.append(nbformat.v4.new_output('display_data',data=rich.data,metadata=rich.metadata))
            cell.outputs=outputs;cell.execution_count=count
        nb.metadata['seasonal_execution']={'scope':'seasonal_lomo tagged cells only; historical outputs retained',
                                           'interpreter':sys.executable,'code_cells':count}
        nbformat.validate(nb);nbformat.write(nb,target)
        print(f'Executed {count} seasonal cells: {path}',flush=True)


if __name__=='__main__': main()
