import os

import numpy as np
import nibabel as nb
from nipype.interfaces.base import File, SimpleInterface, TraitedSpec, traits, CommandLineInputSpec, InputMultiPath
from nipype.utils.filemanip import fname_presuffix

#reusable simple input and output specs
class SimpleMathInputSpec(TraitedSpec):
    in_file = File(exists=True, mandatory=True, desc="Input imaging file")

class SimpleMathOutputSpec(TraitedSpec):
    out_file = File(desc="Output file name")

#binarization with nibabel and numpy
#could replace with niimath
class BinarizeVol(SimpleInterface):
    "Set values less than 0 to 0, greater than 0 to 1"

    input_spec = SimpleMathInputSpec
    output_spec = SimpleMathOutputSpec

    def _run_interface(self, runtime):
        #load in img data
        in_img = nb.load(self.inputs.in_file)
        in_img_data = in_img.get_fdata()

        #set as array for easy broadcasting
        in_img_data = np.array(in_img_data)

        #perform binarization
        in_img_data[in_img_data<0] = 0
        in_img_data[in_img_data>0] = 1

        out_img = nb.Nifti1Image(in_img_data, in_img.affine, header=in_img.header)
        out_file = fname_presuffix(self.inputs.in_file, suffix="_bin", newpath=runtime.cwd)
        out_img.to_filename(out_file)

        self._results["out_file"] = out_file
        return runtime