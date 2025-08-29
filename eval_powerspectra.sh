source .venv/bin/activate
python -m eval_2 --train_filepath "/p/project1/training2533/corradini1/WeGenDiffusion/samples-previous/real/" --gen_filepath "/p/project1/training2533/corradini1/WeGenDiffusion/samples-previous/generated/" --conditional False --num_samples 9 --expid "prevtimestep"

#ppython -m eval_2 --train_filepath "/p/project1/training2533/corradini1/WeGenDiffusion/samples-season/0/" --gen_filepath "/p/project1/training2533/corradini1/WeGenDiffusion/samples-season/0/" --conditional False --num_samples 9 --expid "season_winter"

#ppython -m eval_2 --train_filepath "/p/project1/training2533/patnala1/WeGenDiffusion/data/2011_t2m_era5_2degJJA.nc" --gen_filepath "/p/project1/training2533/corradini1/WeGenDiffusion/samples-season/2/" --conditional False --num_samples 9 --expid "season_summer"

#ppython -m eval_2 --train_filepath "/p/project1/training2533/patnala1/WeGenDiffusion/data/2011_t2m_era5_2degDJF.nc" --gen_filepath "/p/project1/training2533/corradini1/WeGenDiffusion/samples-season/0/" --conditional False --num_samples 9 --expid "season_winter"

#ppython -m eval_2 --train_filepath "/p/project1/training2533/patnala1/WeGenDiffusion/data/months/2011_t2m_era5_2deg01.nc" --gen_filepath "/p/project1/training2533/corradini1/WeGenDiffusion/samples-months/0/" --conditional False --num_samples 9 --expid "months_january"

#ppython -m eval_2 --train_filepath "/p/project1/training2533/patnala1/WeGenDiffusion/data/months/2011_t2m_era5_2deg07.nc" --gen_filepath "/p/project1/training2533/corradini1/WeGenDiffusion/samples-months/6/" --conditional False --num_samples 9 --expid "months_july"
