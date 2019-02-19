## ML-Engine Workflow
task.pyにて行う一連の動作

```mermaid
graph LR

  subgraph GCS

    subgraph data
      data1[train.libfm]
      data2[validation.libfm]
      data3[test.libfm]
    end

    subgraph idx_dict_json
      json1[user_idx_dict.json]
      json2[idx_use_dict.json]
      json3[item_idx_dict.json]
      json4[idx_item_dict.json]
    end

    subgraph params_json
      params.json
    end

    subgraph model
      model.hdf5
    end

    subgraph log
      model_result.csv
      time_result.csv
    end

  end

  subgraph training

    subgraph config
      hptuning_config.yaml
    end

    subgraph training
      LoadData.py
      NeuralFM.py
      test_evaluate.py
      task.py
    end

    subgraph utils
      utils.py
    end
  end

  subgraph hparams
    hparams
  end



  hptuning_config.yaml --> hparams
  hparams --> task.py
  task.py --> LoadData.py

  data1 --> LoadData.py
  data2 --> LoadData.py
  data3 --> LoadData.py


  LoadData.py --> json1
  LoadData.py --> json2
  LoadData.py --> json3
  LoadData.py --> json4
  LoadData.py --> params.json

  LoadData.py --> NeuralFM.py
  utils.py --> NeuralFM.py

  NeuralFM.py --> test_evaluate.py
  NeuralFM.py --> model.hdf5

  test_evaluate.py --> model_result.csv
  test_evaluate.py --> time_result.csv





```
