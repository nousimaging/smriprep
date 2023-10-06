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

class BinaryMathInputSpec(TraitedSpec):
    in_file = File(exists=True, mandatory=True, desc="Image to operate on")
    operand_file = File(exists=True, mandatory=True, desc="Image to perform operation with")
    operand_value = traits.Float(mandatory=False, desc="Value to perform operation with")

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
    
class AddVol(SimpleInterface):

    input_spec = BinaryMathInputSpec
    output_spec = SimpleMathOutputSpec

    def _run_interface(self, runtime):
        #load in img data
        in_img = self.inputs.in_file
        op_file = self.inputs.operand_file
        
        #define output fname
        out_file = fname_presuffix(self.inputs.in_file, suffix="_add", newpath=runtime.cwd)

        #define niimath command string
        cmd_string = 'niimath {in_img} -add {op_img} {outfile}'.format(
            in_img = in_img,
            op_img = op_file,
            outfile=out_file
        )
        
        #call niimath
        os.system(cmd_string)

        self._results["out_file"] = out_file
        return runtime
    
class NM_ThreshBin(SimpleInterface):

    input_spec = SimpleMathInputSpec
    output_spec = SimpleMathOutputSpec

    def _run_interface(self, runtime):
        #load in img data
        in_img = self.inputs.in_file
        
        #define output fname
        out_file = fname_presuffix(self.inputs.in_file, suffix="_thrbin", newpath=runtime.cwd)

        #define niimath command string
        cmd_string = 'niimath {in_img} -thr 0 -bin -mul 255 {outfile}'.format(
            in_img = in_img,
            outfile=out_file
        )
        
        #call niimath
        os.system(cmd_string)

        self._results["out_file"] = out_file
        return runtime
    
class NM_UthreshBin(SimpleInterface):

    input_spec = SimpleMathInputSpec
    output_spec = SimpleMathOutputSpec

    def _run_interface(self, runtime):
        #load in img data
        in_img = self.inputs.in_file
        
        #define output fname
        out_file = fname_presuffix(self.inputs.in_file, suffix="_uthrbin", newpath=runtime.cwd)

        #define niimath command string
        cmd_string = 'niimath {in_img} -uthr 0 -abs -bin -mul 255 {outfile}'.format(
            in_img = in_img,
            outfile=out_file
        )
        
        #call niimath
        os.system(cmd_string)

        self._results["out_file"] = out_file
        return runtime