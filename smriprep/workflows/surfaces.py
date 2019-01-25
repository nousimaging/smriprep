# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Surface preprocessing workflows
-------------------------------

``smriprep`` uses FreeSurfer_ to reconstruct surfaces from T1w/T2w
structural images.

.. autofunction:: init_surface_recon_wf
.. autofunction:: init_autorecon_resume_wf
.. autofunction:: init_gifti_surface_wf

"""
from nipype.pipeline import engine as pe
from nipype.interfaces import (
    io as nio,
    utility as niu,
    freesurfer as fs,
)

from niworkflows.engine.workflows import LiterateWorkflow as Workflow
from niworkflows.interfaces.freesurfer import (
    FSDetectInputs,
    FSInjectBrainExtracted,
    MakeMidthickness,
    PatchedLTAConvert as LTAConvert,
    PatchedRobustRegister as RobustRegister,
    RefineBrainMask,
)
from niworkflows.interfaces.segmentation import ReconAllRPT
from niworkflows.interfaces.surf import NormalizeSurf


def init_surface_recon_wf(omp_nthreads, hires, name='surface_recon_wf'):
    r"""
    This workflow reconstructs anatomical surfaces using FreeSurfer's ``recon-all``.

    Reconstruction is performed in three phases.
    The first phase initializes the subject with T1w and T2w (if available)
    structural images and performs basic reconstruction (``autorecon1``) with the
    exception of skull-stripping.
    For example, a subject with only one session with T1w and T2w images
    would be processed by the following command::

        $ recon-all -sd <output dir>/freesurfer -subjid sub-<subject_label> \
            -i <bids-root>/sub-<subject_label>/anat/sub-<subject_label>_T1w.nii.gz \
            -T2 <bids-root>/sub-<subject_label>/anat/sub-<subject_label>_T2w.nii.gz \
            -autorecon1 \
            -noskullstrip

    The second phase imports an externally computed skull-stripping mask.
    This workflow refines the external brainmask using the internal mask
    implicit the the FreeSurfer's ``aseg.mgz`` segmentation,
    to reconcile ANTs' and FreeSurfer's brain masks.

    First, the ``aseg.mgz`` mask from FreeSurfer is refined in two
    steps, using binary morphological operations:

      1. With a binary closing operation the sulci are included
         into the mask. This results in a smoother brain mask
         that does not exclude deep, wide sulci.

      2. Fill any holes (typically, there could be a hole next to
         the pineal gland and the corpora quadrigemina if the great
         cerebral brain is segmented out).

    Second, the brain mask is grown, including pixels that have a high likelihood
    to the GM tissue distribution:

      3. Dilate and substract the brain mask, defining the region to search for candidate
         pixels that likely belong to cortical GM.

      4. Pixels found in the search region that are labeled as GM by ANTs
         (during ``antsBrainExtraction.sh``) are directly added to the new mask.

      5. Otherwise, estimate GM tissue parameters locally in  patches of ``ww`` size,
         and test the likelihood of the pixel to belong in the GM distribution.

    This procedure is inspired on mindboggle's solution to the problem:
    https://github.com/nipy/mindboggle/blob/7f91faaa7664d820fe12ccc52ebaf21d679795e2/mindboggle/guts/segment.py#L1660


    The final phase resumes reconstruction, using the T2w image to assist
    in finding the pial surface, if available.
    See :py:func:`~smriprep.workflows.surfaces.init_autorecon_resume_wf` for details.


    Memory annotations for FreeSurfer are based off `their documentation
    <https://surfer.nmr.mgh.harvard.edu/fswiki/SystemRequirements>`_.
    They specify an allocation of 4GB per subject. Here we define 5GB
    to have a certain margin.



    .. workflow::
        :graph2use: orig
        :simple_form: yes

        from smriprep.workflows.surfaces import init_surface_recon_wf
        wf = init_surface_recon_wf(omp_nthreads=1, hires=True)

    **Parameters**

        omp_nthreads : int
            Maximum number of threads an individual process may use
        hires : bool
            Enable sub-millimeter preprocessing in FreeSurfer

    **Inputs**

        t1w
            List of T1-weighted structural images
        t2w
            List of T2-weighted structural images (only first used)
        flair
            List of FLAIR images
        skullstripped_t1
            Skull-stripped T1-weighted image (or mask of image)
        ants_segs
            Brain tissue segmentation from ANTS ``antsBrainExtraction.sh``
        corrected_t1
            INU-corrected, merged T1-weighted image
        subjects_dir
            FreeSurfer SUBJECTS_DIR
        subject_id
            FreeSurfer subject ID

    **Outputs**

        subjects_dir
            FreeSurfer SUBJECTS_DIR
        subject_id
            FreeSurfer subject ID
        t1_2_fsnative_forward_transform
            LTA-style affine matrix translating from T1w to FreeSurfer-conformed subject space
        t1_2_fsnative_reverse_transform
            LTA-style affine matrix translating from FreeSurfer-conformed subject space to T1w
        surfaces
            GIFTI surfaces for gray/white matter boundary, pial surface,
            midthickness (or graymid) surface, and inflated surfaces
        out_brainmask
            Refined brainmask, derived from FreeSurfer's ``aseg`` volume
        out_aseg
            FreeSurfer's aseg segmentation, in native T1w space
        out_aparc
            FreeSurfer's aparc+aseg segmentation, in native T1w space
        out_report
            Reportlet visualizing quality of surface alignment

    **Subworkflows**

        * :py:func:`~smriprep.workflows.surfaces.init_autorecon_resume_wf`
        * :py:func:`~smriprep.workflows.surfaces.init_gifti_surface_wf`
    """

    workflow = Workflow(name=name)
    workflow.__desc__ = """\
Brain surfaces were reconstructed using `recon-all` [FreeSurfer {fs_ver},
RRID:SCR_001847, @fs_reconall], and the brain mask estimated
previously was refined with a custom variation of the method to reconcile
ANTs-derived and FreeSurfer-derived segmentations of the cortical
gray-matter of Mindboggle [RRID:SCR_002438, @mindboggle].
""".format(fs_ver=fs.Info().looseversion() or '<ver>')

    inputnode = pe.Node(
        niu.IdentityInterface(
            fields=['t1w', 't2w', 'flair', 'skullstripped_t1', 'corrected_t1', 'ants_segs',
                    'subjects_dir', 'subject_id']), name='inputnode')
    outputnode = pe.Node(
        niu.IdentityInterface(
            fields=['subjects_dir', 'subject_id', 't1_2_fsnative_forward_transform',
                    't1_2_fsnative_reverse_transform', 'surfaces', 'out_brainmask',
                    'out_aseg', 'out_aparc', 'out_report']),
        name='outputnode')

    recon_config = pe.Node(FSDetectInputs(hires_enabled=hires), name='recon_config')

    fov_check = pe.Node(niu.Function(function=_check_cw256), name='fov_check')

    autorecon1 = pe.Node(
        fs.ReconAll(directive='autorecon1', openmp=omp_nthreads),
        name='autorecon1', n_procs=omp_nthreads, mem_gb=5)
    autorecon1.interface._can_resume = False
    autorecon1.interface._always_run = True

    skull_strip_extern = pe.Node(FSInjectBrainExtracted(), name='skull_strip_extern')

    fsnative_2_t1_xfm = pe.Node(RobustRegister(auto_sens=True, est_int_scale=True),
                                name='fsnative_2_t1_xfm')
    t1_2_fsnative_xfm = pe.Node(LTAConvert(out_lta=True, invert=True),
                                name='t1_2_fsnative_xfm')

    autorecon_resume_wf = init_autorecon_resume_wf(omp_nthreads=omp_nthreads)
    gifti_surface_wf = init_gifti_surface_wf()

    aseg_to_native_wf = init_segs_to_native_wf()
    aparc_to_native_wf = init_segs_to_native_wf(segmentation='aparc_aseg')
    refine = pe.Node(RefineBrainMask(), name='refine')

    workflow.connect([
        # Configuration
        (inputnode, recon_config, [('t1w', 't1w_list'),
                                   ('t2w', 't2w_list'),
                                   ('flair', 'flair_list')]),
        # Passing subjects_dir / subject_id enforces serial order
        (inputnode, autorecon1, [('subjects_dir', 'subjects_dir'),
                                 ('subject_id', 'subject_id')]),
        (autorecon1, skull_strip_extern, [('subjects_dir', 'subjects_dir'),
                                          ('subject_id', 'subject_id')]),
        (skull_strip_extern, autorecon_resume_wf, [('subjects_dir', 'inputnode.subjects_dir'),
                                                   ('subject_id', 'inputnode.subject_id')]),
        (autorecon_resume_wf, gifti_surface_wf, [
            ('outputnode.subjects_dir', 'inputnode.subjects_dir'),
            ('outputnode.subject_id', 'inputnode.subject_id')]),
        # Reconstruction phases
        (inputnode, autorecon1, [('t1w', 'T1_files')]),
        (inputnode, fov_check, [('t1w', 'in_files')]),
        (fov_check, autorecon1, [('out', 'flags')]),
        (recon_config, autorecon1, [('t2w', 'T2_file'),
                                    ('flair', 'FLAIR_file'),
                                    ('hires', 'hires'),
                                    # First run only (recon-all saves expert options)
                                    ('mris_inflate', 'mris_inflate')]),
        (inputnode, skull_strip_extern, [('skullstripped_t1', 'in_brain')]),
        (recon_config, autorecon_resume_wf, [('use_t2w', 'inputnode.use_T2'),
                                             ('use_flair', 'inputnode.use_FLAIR')]),
        # Construct transform from FreeSurfer conformed image to sMRIPrep
        # reoriented image
        (inputnode, fsnative_2_t1_xfm, [('t1w', 'target_file')]),
        (autorecon1, fsnative_2_t1_xfm, [('T1', 'source_file')]),
        (fsnative_2_t1_xfm, gifti_surface_wf, [
            ('out_reg_file', 'inputnode.t1_2_fsnative_reverse_transform')]),
        (fsnative_2_t1_xfm, t1_2_fsnative_xfm, [('out_reg_file', 'in_lta')]),
        # Refine ANTs mask, deriving new mask from FS' aseg
        (inputnode, refine, [('corrected_t1', 'in_anat'),
                             ('ants_segs', 'in_ants')]),
        (inputnode, aseg_to_native_wf, [('corrected_t1', 'inputnode.in_file')]),
        (autorecon_resume_wf, aseg_to_native_wf, [
            ('outputnode.subjects_dir', 'inputnode.subjects_dir'),
            ('outputnode.subject_id', 'inputnode.subject_id')]),
        (inputnode, aparc_to_native_wf, [('corrected_t1', 'inputnode.in_file')]),
        (autorecon_resume_wf, aparc_to_native_wf, [
            ('outputnode.subjects_dir', 'inputnode.subjects_dir'),
            ('outputnode.subject_id', 'inputnode.subject_id')]),
        (aseg_to_native_wf, refine, [('outputnode.out_file', 'in_aseg')]),

        # Output
        (autorecon_resume_wf, outputnode, [('outputnode.subjects_dir', 'subjects_dir'),
                                           ('outputnode.subject_id', 'subject_id'),
                                           ('outputnode.out_report', 'out_report')]),
        (gifti_surface_wf, outputnode, [('outputnode.surfaces', 'surfaces')]),
        (t1_2_fsnative_xfm, outputnode, [('out_lta', 't1_2_fsnative_forward_transform')]),
        (fsnative_2_t1_xfm, outputnode, [('out_reg_file', 't1_2_fsnative_reverse_transform')]),
        (refine, outputnode, [('out_file', 'out_brainmask')]),
        (aseg_to_native_wf, outputnode, [('outputnode.out_file', 'out_aseg')]),
        (aparc_to_native_wf, outputnode, [('outputnode.out_file', 'out_aparc')]),
    ])

    return workflow


def init_autorecon_resume_wf(omp_nthreads, name='autorecon_resume_wf'):
    r"""
    This workflow resumes recon-all execution, assuming the `-autorecon1` stage
    has been completed.

    In order to utilize resources efficiently, this is broken down into five
    sub-stages; after the first stage, the second and third stages may be run
    simultaneously, and the fourth and fifth stages may be run simultaneously,
    if resources permit::

        $ recon-all -sd <output dir>/freesurfer -subjid sub-<subject_label> \
            -autorecon2-volonly
        $ recon-all -sd <output dir>/freesurfer -subjid sub-<subject_label> \
            -autorecon-hemi lh \
            -noparcstats -nocortparc2 -noparcstats2 -nocortparc3 \
            -noparcstats3 -nopctsurfcon -nohyporelabel -noaparc2aseg \
            -noapas2aseg -nosegstats -nowmparc -nobalabels
        $ recon-all -sd <output dir>/freesurfer -subjid sub-<subject_label> \
            -autorecon-hemi rh \
            -noparcstats -nocortparc2 -noparcstats2 -nocortparc3 \
            -noparcstats3 -nopctsurfcon -nohyporelabel -noaparc2aseg \
            -noapas2aseg -nosegstats -nowmparc -nobalabels
        $ recon-all -sd <output dir>/freesurfer -subjid sub-<subject_label> \
            -autorecon3 -hemi lh -T2pial
        $ recon-all -sd <output dir>/freesurfer -subjid sub-<subject_label> \
            -autorecon3 -hemi rh -T2pial

    The excluded steps in the second and third stages (``-no<option>``) are not
    fully hemisphere independent, and are therefore postponed to the final two
    stages.

    .. workflow::
        :graph2use: orig
        :simple_form: yes

        from smriprep.workflows.surfaces import init_autorecon_resume_wf
        wf = init_autorecon_resume_wf(omp_nthreads=1)

    **Inputs**

        subjects_dir
            FreeSurfer SUBJECTS_DIR
        subject_id
            FreeSurfer subject ID
        use_T2
            Refine pial surface using T2w image
        use_FLAIR
            Refine pial surface using FLAIR image

    **Outputs**

        subjects_dir
            FreeSurfer SUBJECTS_DIR
        subject_id
            FreeSurfer subject ID
        out_report
            Reportlet visualizing quality of surface alignment

    """
    workflow = Workflow(name=name)

    inputnode = pe.Node(
        niu.IdentityInterface(
            fields=['subjects_dir', 'subject_id', 'use_T2', 'use_FLAIR']),
        name='inputnode')

    outputnode = pe.Node(
        niu.IdentityInterface(
            fields=['subjects_dir', 'subject_id', 'out_report']),
        name='outputnode')

    autorecon2_vol = pe.Node(
        fs.ReconAll(directive='autorecon2-volonly', openmp=omp_nthreads),
        n_procs=omp_nthreads, mem_gb=5, name='autorecon2_vol')
    autorecon2_vol.interface._always_run = True

    autorecon_surfs = pe.MapNode(
        fs.ReconAll(
            directive='autorecon-hemi',
            flags=['-noparcstats', '-nocortparc2', '-noparcstats2',
                   '-nocortparc3', '-noparcstats3', '-nopctsurfcon',
                   '-nohyporelabel', '-noaparc2aseg', '-noapas2aseg',
                   '-nosegstats', '-nowmparc', '-nobalabels'],
            openmp=omp_nthreads),
        iterfield='hemi', n_procs=omp_nthreads, mem_gb=5,
        name='autorecon_surfs')
    autorecon_surfs.inputs.hemi = ['lh', 'rh']
    autorecon_surfs.interface._always_run = True

    autorecon3 = pe.MapNode(
        fs.ReconAll(directive='autorecon3', openmp=omp_nthreads),
        iterfield='hemi', n_procs=omp_nthreads, mem_gb=5,
        name='autorecon3')
    autorecon3.inputs.hemi = ['lh', 'rh']
    autorecon3.interface._always_run = True

    # Only generate the report once; should be nothing to do
    recon_report = pe.Node(
        ReconAllRPT(directive='autorecon3', generate_report=True),
        name='recon_report', mem_gb=5)
    recon_report.interface._always_run = True

    def _dedup(in_list):
        vals = set(in_list)
        if len(vals) > 1:
            raise ValueError(
                "Non-identical values can't be deduplicated:\n{!r}".format(in_list))
        return vals.pop()

    workflow.connect([
        (inputnode, autorecon3, [('use_T2', 'use_T2'),
                                 ('use_FLAIR', 'use_FLAIR')]),
        (inputnode, autorecon2_vol, [('subjects_dir', 'subjects_dir'),
                                     ('subject_id', 'subject_id')]),
        (autorecon2_vol, autorecon_surfs, [('subjects_dir', 'subjects_dir'),
                                           ('subject_id', 'subject_id')]),
        (autorecon_surfs, autorecon3, [(('subjects_dir', _dedup), 'subjects_dir'),
                                       (('subject_id', _dedup), 'subject_id')]),
        (autorecon3, outputnode, [(('subjects_dir', _dedup), 'subjects_dir'),
                                  (('subject_id', _dedup), 'subject_id')]),
        (autorecon3, recon_report, [(('subjects_dir', _dedup), 'subjects_dir'),
                                    (('subject_id', _dedup), 'subject_id')]),
        (recon_report, outputnode, [('out_report', 'out_report')]),
    ])

    return workflow


def init_gifti_surface_wf(name='gifti_surface_wf'):
    r"""
    This workflow prepares GIFTI surfaces from a FreeSurfer subjects directory

    If midthickness (or graymid) surfaces do not exist, they are generated and
    saved to the subject directory as ``lh/rh.midthickness``.
    These, along with the gray/white matter boundary (``lh/rh.smoothwm``), pial
    sufaces (``lh/rh.pial``) and inflated surfaces (``lh/rh.inflated``) are
    converted to GIFTI files.
    Additionally, the vertex coordinates are :py:class:`recentered
    <smriprep.interfaces.NormalizeSurf>` to align with native T1w space.

    .. workflow::
        :graph2use: orig
        :simple_form: yes

        from smriprep.workflows.surfaces import init_gifti_surface_wf
        wf = init_gifti_surface_wf()

    **Inputs**

        subjects_dir
            FreeSurfer SUBJECTS_DIR
        subject_id
            FreeSurfer subject ID
        t1_2_fsnative_reverse_transform
            LTA formatted affine transform file (inverse)

    **Outputs**

        surfaces
            GIFTI surfaces for gray/white matter boundary, pial surface,
            midthickness (or graymid) surface, and inflated surfaces

    """
    workflow = Workflow(name=name)

    inputnode = pe.Node(niu.IdentityInterface(['subjects_dir', 'subject_id',
                                               't1_2_fsnative_reverse_transform']),
                        name='inputnode')
    outputnode = pe.Node(niu.IdentityInterface(['surfaces']), name='outputnode')

    get_surfaces = pe.Node(nio.FreeSurferSource(), name='get_surfaces')

    midthickness = pe.MapNode(
        MakeMidthickness(thickness=True, distance=0.5, out_name='midthickness'),
        iterfield='in_file',
        name='midthickness')

    save_midthickness = pe.Node(nio.DataSink(parameterization=False),
                                name='save_midthickness')

    surface_list = pe.Node(niu.Merge(4, ravel_inputs=True),
                           name='surface_list', run_without_submitting=True)
    fs_2_gii = pe.MapNode(fs.MRIsConvert(out_datatype='gii'),
                          iterfield='in_file', name='fs_2_gii')
    fix_surfs = pe.MapNode(NormalizeSurf(), iterfield='in_file', name='fix_surfs')

    workflow.connect([
        (inputnode, get_surfaces, [('subjects_dir', 'subjects_dir'),
                                   ('subject_id', 'subject_id')]),
        (inputnode, save_midthickness, [('subjects_dir', 'base_directory'),
                                        ('subject_id', 'container')]),
        # Generate midthickness surfaces and save to FreeSurfer derivatives
        (get_surfaces, midthickness, [('smoothwm', 'in_file'),
                                      ('graymid', 'graymid')]),
        (midthickness, save_midthickness, [('out_file', 'surf.@graymid')]),
        # Produce valid GIFTI surface files (dense mesh)
        (get_surfaces, surface_list, [('smoothwm', 'in1'),
                                      ('pial', 'in2'),
                                      ('inflated', 'in3')]),
        (save_midthickness, surface_list, [('out_file', 'in4')]),
        (surface_list, fs_2_gii, [('out', 'in_file')]),
        (fs_2_gii, fix_surfs, [('converted', 'in_file')]),
        (inputnode, fix_surfs, [('t1_2_fsnative_reverse_transform', 'transform_file')]),
        (fix_surfs, outputnode, [('out_file', 'surfaces')]),
    ])
    return workflow


def init_segs_to_native_wf(name='segs_to_native', segmentation='aseg'):
    """
    Get a segmentation from FreeSurfer conformed space into native T1w space


    .. workflow::
        :graph2use: orig
        :simple_form: yes

        from smriprep.workflows.surfaces import init_segs_to_native_wf
        wf = init_segs_to_native_wf()


    **Parameters**
        segmentation
            The name of a segmentation ('aseg' or 'aparc_aseg' or 'wmparc')

    **Inputs**

        in_file
            Anatomical, merged T1w image after INU correction
        subjects_dir
            FreeSurfer SUBJECTS_DIR
        subject_id
            FreeSurfer subject ID


    **Outputs**

        out_file
            The selected segmentation, after resampling in native space
    """
    workflow = Workflow(name='%s_%s' % (name, segmentation))
    inputnode = pe.Node(niu.IdentityInterface([
        'in_file', 'subjects_dir', 'subject_id']), name='inputnode')
    outputnode = pe.Node(niu.IdentityInterface(['out_file']), name='outputnode')
    # Extract the aseg and aparc+aseg outputs
    fssource = pe.Node(nio.FreeSurferSource(), name='fs_datasource')
    tonative = pe.Node(fs.Label2Vol(), name='tonative')
    tonii = pe.Node(fs.MRIConvert(out_type='niigz', resample_type='nearest'), name='tonii')

    if segmentation.startswith('aparc'):
        if segmentation == 'aparc_aseg':
            def _sel(x): return [parc for parc in x if 'aparc+' in parc][0]
        elif segmentation == 'aparc_a2009s':
            def _sel(x): return [parc for parc in x if 'a2009s+' in parc][0]
        elif segmentation == 'aparc_dkt':
            def _sel(x): return [parc for parc in x if 'DKTatlas+' in parc][0]
        segmentation = (segmentation, _sel)

    workflow.connect([
        (inputnode, fssource, [
            ('subjects_dir', 'subjects_dir'),
            ('subject_id', 'subject_id')]),
        (inputnode, tonii, [('in_file', 'reslice_like')]),
        (fssource, tonative, [(segmentation, 'seg_file'),
                              ('rawavg', 'template_file'),
                              ('aseg', 'reg_header')]),
        (tonative, tonii, [('vol_label_file', 'in_file')]),
        (tonii, outputnode, [('out_file', 'out_file')]),
    ])
    return workflow


def _check_cw256(in_files):
    from nibabel.funcs import concat_images
    if isinstance(in_files, str):
        in_files = [in_files]
    if any((s > 256 for s in concat_images(in_files).shape[:3])):
        return ['-noskullstrip', '-cw256']
    return '-noskullstrip'
