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
    
class ThreshBin(SimpleInterface):
    input_spec = SimpleMathInputSpec
    output_spec = SimpleMathOutputSpec

    def _run_interface(self,runtime):
        #read in img
        in_img = nb.load(self.inputs.in_file)

        #load data as array
        in_data = np.array(in_img.get_fdata())

        #match math command  -thr 0 -bin -mul 255

        #apply threshold (0 anything below 0)
        in_data[in_data<0] = 0

        #binarize
        #in_data[in_data<0] = 0
        in_data[in_data>0] = 1

        #multiply by 255
        in_data = in_data * 255

        #write out new img
        out_img = nb.Nifti1Image(in_data, in_img.affine, header=in_img.header)
        out_file = fname_presuffix(self.inputs.in_file, suffix="_thrbin", newpath=runtime.cwd)
        out_img.to_filename(out_file)

        self._results["out_file"] = out_file
        return runtime

class UThreshBin(SimpleInterface):

    input_spec = SimpleMathInputSpec
    output_spec = SimpleMathOutputSpec

    def _run_interface(self,runtime):
        #read in img
        in_img = nb.load(self.inputs.in_file)

        #load data as array
        in_data = np.array(in_img.get_fdata())

        #match math command  -uthr 0 -abs -bin -mul 255

        #apply upper threshold (0 anything above 0)
        in_data[in_data>0] = 0

        #absolute value
        in_data = np.absolute(in_data)

        #binarize
        in_data[in_data<0] = 0 #should be redundant
        in_data[in_data>0] = 1

        #multiply by 255
        in_data = in_data * 255

        #write out new img
        out_img = nb.Nifti1Image(in_data, in_img.affine, header=in_img.header)
        out_file = fname_presuffix(self.inputs.in_file, suffix="_uthrbin", newpath=runtime.cwd)
        out_img.to_filename(out_file)

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
            outfile = out_file
        )
        
        #call niimath
        os.system(cmd_string)

        self._results["out_file"] = out_file
        return runtime
    
class NM_MakeRibbon(SimpleInterface):

    input_spec = BinaryMathInputSpec
    output_spec = SimpleMathOutputSpec

    def _run_interface(self, runtime):
        #load in img data
        in_img = self.inputs.in_file
        op_img = self.inputs.operand_file
        
        #define output fname
        out_file = fname_presuffix(self.inputs.in_file, suffix="_ribbon", newpath=runtime.cwd)

        #define niimath command string
        cmd_string = 'niimath {in_img} -mas {op_img} -mul 255 {outfile}'.format(
            in_img = in_img,
            op_img = op_img,
            outfile = out_file
        )
        
        #call niimath
        os.system(cmd_string)

        self._results["out_file"] = out_file
        return runtime
    
class CustomApplyMaskInputSpec(TraitedSpec):
    in_file = File(
        exists=True,
        mandatory=True,
        desc="Image to be masked")
    mask_file = File(
        exists=True,
        mandatory=True,
        desc='Mask to be applied')

class CustomApplyMaskOutputSpec(TraitedSpec):
    out_file = File(exist=True, desc="Image with mask applied")

class CustomApplyMask(SimpleInterface):
    input_spec = CustomApplyMaskInputSpec
    output_spec = CustomApplyMaskOutputSpec

    def _run_interface(self, runtime):
        #define masked output name
        out_file = fname_presuffix(
            self.inputs.in_file,
            newpath=runtime.cwd,
            suffix='_masked.nii.gz',
            use_ext=False)

        #load in input and mask
        input_img = nb.load(self.inputs.in_file)
        input_data = input_img.get_fdata()
        mask_data = nb.load(self.inputs.mask_file).get_fdata()
        #elementwise multiplication to apply mask
        out_data = input_data * mask_data
        #save out masked image and pass on file name
        nb.Nifti1Image(out_data, input_img.affine, header=input_img.header).to_filename(out_file)
        self._results['out_file'] = out_file
        return runtime