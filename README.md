# Post-FairGAN
Run the following line to apply post-fairgan to synthetic data in the specified 'directory' with respect to the provided protected feature:  
`python main.py [directory] [dataset_name] [training_file] [protected feature] [privileged value] [predicted feature] [preferred value] [selected_data_percentage (ex. .5)]`

## Example
`python main.py example/german-synthetic/ german example/GERMAN-SPLIT-TRAIN-60.csv gender female labels 1 .5`
