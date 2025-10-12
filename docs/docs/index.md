# Obesity Estimation Equipo53 documentation!

## Description

Este proyecto servirá para poner en práctica la metodología de MLOps para desarrollo de proyectos de Machine Learning, el cual se dividirá en distintas fases las cuales constarán de las distintas etapas de un producto de ML.

## Commands

The Makefile contains the central entry points for common tasks related to this project.

### Syncing data to cloud storage

* `make sync_data_up` will use `aws s3 sync` to recursively sync files in `data/` up to `s3://bucket-name/data/`.
* `make sync_data_down` will use `aws s3 sync` to recursively sync files from `s3://bucket-name/data/` to `data/`.


