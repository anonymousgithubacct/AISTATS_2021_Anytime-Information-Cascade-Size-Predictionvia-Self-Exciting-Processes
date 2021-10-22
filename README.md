# Anytime Information Cascade Size Prediction via Self-Exciting Processes

This repository hosts code for the paper recently submitted to AISTATS 2022 titled "Anytime Information Cascade Size Prediction via Self-Exciting Processes".

## Running the code

We have a easily runnable example at main.py which can simply be run via the following python command. This code is designed to train on an observed retweet sequence upto censoring time $t_c$. You can run the example cascade found in `data/seismic/example_cascade_2.json` using the following command

```
python3 main.py --run_example=True
```

If you want to run with custom $t_c$, data file and/or results output path, you can make use of the following command line argument.

```
python3 main.py --data_file_path=**include_file_name** --censoring_time="1h" --output_path=**include_output_path**
```


## Data considerations

In order to use different realization files, it should be saved as a `json` file with the following format:

```
{ 'cid': **cascade ID**,
  'post_time_day': ** Absolute time of posting of tweet ** 
  'hw': list of events, where each event is a list of time and mark for that event, i.e., [[$t_0, m_0$], [$t_1, m_1$], [$t_2, m_2$], ...]
}
```


## Citing our work
This work is still under review at AISTATS 2022 and has double-blind reviews, as a result, we cannot yet post a bibliography for any citation purposes. We ask that you contact the author of the repository directly.
