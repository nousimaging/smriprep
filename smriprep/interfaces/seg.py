import os
import numpy as np
import nibabel as nb
from nipype.interfaces.base import File, SimpleInterface, TraitedSpec, traits, CommandLineInputSpec, InputMultiPath
from nipype.utils.filemanip import fname_presuffix

class RelabelAsegInputSpec(TraitedSpec):
    in_aseg = File(exists=True, mandatory=True, desc="Freesurfer aseg nifti")

class RelabelAsegOutputSpec(TraitedSpec):
    out_file = File(desc="Output file name")

class RelabelAseg(SimpleInterface):
    input_spec = RelabelAsegInputSpec
    output_spec = RelabelAsegOutputSpec

    def _run_interface(self, runtime):
        #get new lut
        aseg_lut = _aseg_to_three()
        aseg_lut = np.array(aseg_lut, dtype="int16")

        #define outname
        out_file = fname_presuffix(self.inputs.in_aseg, suffix="_asegrelabel", newpath=runtime.cwd)

        #load in data
        segm = nb.load(self.inputs.in_aseg)
        hdr = segm.header.copy()
        hdr.set_data_dtype("int16")

        #relabel and save back out
        segm.__class__(
            aseg_lut[np.asanyarray(segm.dataobj, dtype=int)].astype("int16"), segm.affine, hdr
        ).to_filename(out_file)

        return runtime

class SplitAsegOutputSpec(TraitedSpec):
    out_gm = File(desc="GM mask")
    out_wm = File(desc="WM mask")
    out_csf = File(desc="CSF mask")

class SplitAseg(SimpleInterface):
    input_spec = RelabelAsegInputSpec
    output_spec = SplitAsegOutputSpec

    def _run_interface(self, runtime):

        #define outnames
        gm_outname = fname_presuffix(self.inputs.in_aseg, suffix="_gmseg", newpath=runtime.cwd)
        wm_outname = fname_presuffix(self.inputs.in_aseg, suffix="_wmseg", newpath=runtime.cwd)
        csf_outname = fname_presuffix(self.inputs.in_aseg, suffix="_csfseg", newpath=runtime.cwd)

        #load in aseg data
        in_img = nb.load(self.inputs.in_aseg)
        in_img_data = in_img.get_fdata()

        #set as array for easy broadcasting
        in_img_data = np.array(in_img_data)

        #init masks
        gm_data = in_img_data.copy()
        wm_data = in_img_data.copy()
        csf_data = in_img_data.copy()

        #extract masks and binarize
        gm_data[gm_data!=1] = 0

        wm_data[wm_data!=2] = 0
        wm_data[wm_data==2] = 1

        csf_data[csf_data!=3] = 0
        csf_data[csf_data==3] = 1

        #write out split segmented images
        out_gm = nb.Nifti1Image(gm_data, in_img.affine, header=in_img.header)
        out_wm = nb.Nifti1Image(wm_data, in_img.affine, header=in_img.header)
        out_csf = nb.Nifti1Image(csf_data, in_img.affine, header=in_img.header)

        out_gm.to_filename(gm_outname)
        out_wm.to_filename(wm_outname)
        out_csf.to_filename(csf_outname)

        self._results["out_gm"] = gm_outname
        self._results["out_wm"] = wm_outname
        self._results["out_csf"] = csf_outname

        return runtime

def _aseg_to_three():
    """
    Map FreeSurfer's segmentation onto a brain (3-)tissue segmentation.

    This function generates an index of 255+0 labels and maps them into zero (bg),
    1 (GM), 2 (WM), or 3 (CSF). The new values are set according to BIDS-Derivatives.
    Then the index is populated (e.g., label 3 in the original segmentation maps to label
    1 in the output).
    The `aseg lookup table
    <https://github.com/freesurfer/freesurfer/blob/2beb96c6099d96508246c14a24136863124566a3/distribution/ASegStatsLUT.txt>`__
    is available in the FreeSurfer source.

    """
    import numpy as np

    # Base struct
    aseg_lut = np.zeros((256,), dtype="int")
    # GM
    aseg_lut[3] = 1
    aseg_lut[8:14] = 1
    aseg_lut[17:21] = 1
    aseg_lut[26:40] = 1
    aseg_lut[42] = 1
    aseg_lut[47:73] = 1

    # CSF
    aseg_lut[4:6] = 3
    aseg_lut[14:16] = 3
    aseg_lut[24] = 3
    aseg_lut[43:45] = 3
    aseg_lut[72] = 3

    # WM
    aseg_lut[2] = 2
    aseg_lut[7] = 2
    aseg_lut[16] = 2
    aseg_lut[28] = 2
    aseg_lut[41] = 2
    aseg_lut[46] = 2
    aseg_lut[60] = 2
    aseg_lut[77:80] = 2
    aseg_lut[250:256] = 2
    return tuple(aseg_lut)