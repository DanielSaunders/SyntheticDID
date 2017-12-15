# SyntheticDID

Generate simulated handwritten texts to use as supplementary data for machine
learning tasks.

## Requirements

* OpenCV
* Python 3
* Java JRE

## Usage

In order to generate synthetic images, a few prerequisites are necessary.
These scripts depend on access to a set of files:

1. Background images (i.e. the pages the text will be placed on)
1. Handwriting samples
1. Stains to place on the pages

By default, these need to be placed in `background_images/`, `handwriting_images/`,
and `stain_imges/` folders relative to the root of this repository.

To get started, some good sources to get some representative data for each data
type follow.

#### Background Images

#### Handwriting Samples

A good dataset to use is the
[IAM Handwriting Database](http://www.fki.inf.unibe.ch/databases/iam-handwriting-database)
which is available free for non-commercial research usage. Note that
downloading the database will require registration.

**N.B.** All handwriting samples should be black text on a white background

#### Stains

A good collection of stains is provided by
[DIVADid](https://diuf.unifr.ch/main/hisdoc/divadid-document-image-degradation)
itself. The small stains can be downloaded
[here](http://diuf.unifr.ch/diva/divadid/spots.zip) and larger stains can be
downloaded [here](http://diuf.unifr.ch/diva/divadid/surfaces.zip)

### Generating Synthetic Images

Once the required data files are in place, a simple demonstration of running
the script is

```bash
./generate_images.py 10
```

which will generate 10 synthetic images in a folder (by default in `tmp/`).

Options can be specified by editing the `options.ini` file or passed in on
the command line. For example,

```bash
./generate_images.py --output_dir=~/synthetic_images 10
```

will generate 10 images and save them in the `~/synthetic_images` directory.

## Explanation of Process

There are three high-level steps to the process of generating these synthetic
images.

1. Add degradations to paper images
1. Alpha blend text to paper documents
1. Further degrade images

  All degradation is done with
  [DIVADid](https://diuf.unifr.ch/main/hisdoc/divadid-document-image-degradation).