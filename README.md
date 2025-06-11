# 2D_hydrodynamic

python setup.py build_ext --inplace

## ???????
### ?windows
set OMP_NUM_THREADS=4
python src/hydro_model/run_simulation.py
### ?Linux?macOS??
export OMP_NUM_THREADS=4
python src/hydro_model/run_simulation.py