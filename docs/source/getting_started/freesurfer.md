To perform inverse modeling, or also called source estimation or source
localization, we need to ensure that a couple prerequisities are met.
Essentially, starting from a collection of 2-dimensional MRI images of coronal,
axial, and saggital slices of a participant's head, we need to construct a
3-dimensional representation of the brain, skull, and scalp. Furthermore,
it's highly advantegous to attach labels to different brain areas according
to common anatomical atlases, so that we could, for example, restrict
subsequent analyses to specific cortical regions, and compare activation
in these regions across participants.

BIDS raw datasets, however, do **not** include any of these 3D representations
and parcellations. (Note that, however, these *derivatives* **are** sometimes
distributed along with a datasets inside a `derivatives/` folder). Instead,
they ship e.g. with T1-weighted images only (and, sometimes, include FLASH
images too).

## Install FreeSurfer

Before running the source-analysis parts of the Study Template, you need to
create said 3D surfaces and parcellations. This is done using the tool
[FreeSurfer](https://surfer.nmr.mgh.harvard.edu/fswiki/). FreeSurfer is a
free software package that runs on macOS and Linux.

To install FreeSurfer 6, follow the [official download and installation
nstructions](https://surfer.nmr.mgh.harvard.edu/fswiki/rel6downloads).

!!! info
    The only currently supported FreeSurfer version is 6.0.

!!! warning
    FreeSurfer does not natively run on Windows. We are currently working on
    ways to make it possible to use it on Windows, too.

!!! warning
    FreeSurfer cannot currently be used on macOS Big Sur. We are working on
    a solution.

## Generate surfaces and brain parcellation

The Study Template provides a convenient way to invoke FreeSurfer. After
[adjusting your configuration file](basic_usage.md#adjust-your-configuration-file),
invoke FreeSurfer via the `run.py` script in the following way:

```shell
python run.py freesurfer --config=/path/to/your/custom_config.py
```

This will run the
[`recon-all` command](https://surfer.nmr.mgh.harvard.edu/fswiki/recon-all)
to create the required surfaces.

!!! info
    This process is very computationally expensive, and will usually take
    several hours to complete. It's a good idea to let this command run
    over night.

## Run source-level analyses

Now you are ready to run the Study Template, including all parts of inverse
modeling. To perform the projection, MNE-Python will first need to detect
brain, skull, and skin, so it can then start constructing the actual BEM
conductor model. These BEM surfaces can be created based on FLASH MRI
(best option) or T1-weighted MRI images (second-best). See the respective
[configuration options](../settings/source/bem.md) to control BEM surface
creation.

*[FLASH]: Fast low angle shot
*[MRI]: Magnetic resonance imaging
*[BEM]: Boundary element model
