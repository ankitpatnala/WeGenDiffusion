

source .venv/bin/activate
python -m eval --val_filepath "./data/2012_t2m_era5_4months_2deg.nc" --gen_filepath "/p/project1/training2533/corradini1/WeGenDiffusion/samples/" --num_samples 2 > log1.txt 2>&1 &
wait

python -m eval --val_filepath "./data/2012_t2m_era5_4months_2deg.nc" --gen_filepath "/p/project1/training2533/corradini1/WeGenDiffusion/samples-season/" --num_samples 2 --conditional True --label 'season' > log2.txt 2>&1 &
wait

python -m eval --val_filepath "./data/2012_t2m_era5_4months_2deg.nc" --gen_filepath "/p/project1/training2533/corradini1/WeGenDiffusion/samples-year/" --num_samples 2 --conditional True --label 'year' > log3.txt 2>&1 &