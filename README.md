# Bayesian Network Structure Estimation

This project aims to estimate the structure of Bayesian Networks using two methods: ScoreBase and OrderBase.

## Usage

1. Clone the repository:
    ```sh
    git clone https://github.com/otyanokosaisai/BayesianNetwork.git
    ```
2. Rewrite compose file if necessary:
    ```
    version: '3.8'
    services:
    basian:
        image: rust_box:latest
        build:
            context: .
            dockerfile: Dockerfile
        ports:
        - "8003:8080"
        volumes:
        - .:/workspace
        command: tail /dev/null
        deploy:
        resources:
            limits:
            cpus: '8'
            memory: 32G
            reservations:
            cpus: '1'
            memory: 512M  

    ```
3. start docker container and access the container:
    ```sh
    docker compose up -d
    ```

4. Edit the configuration files:
    - `setting/exm/oreder.yml` for one dataset examination
    - `setting/exm/score.yml` for one dataset examination
    - `setting/exp/order.yml` for many datasets experiment
    - `setting/exp/score.yml` for many datasets experiment
    - `setting/est/order.yml` for one dataset estimation whose you have no answer
    - `setting/est/score.yml` for one dataset estimation whose you have no answer

    Example configuration file (`setting/exm/oreder.yml`):
    ```yaml
        method: "bdeu"
        bdeu_ess: 1.0
        data_path: "data/asia.csv"
        compare_network_path: "data/asia.dot"
        saving_dir: "results/exm/order"
    ```

    Example configuration file (`exp/order.yml`):
    ```yaml
        method: "bdeu" #(aic, bic, bdeu)
        bdeu_ess: 10.0 #(for bdeu)
        data_dir: "./data" #(data dir)
        compare_network_dir: "./data" #(ans dot file dir)
        data_names:
        - "dataset1" #(dataset name ({dataset}.csv))
        - "dataset2"
        timeout_hours: 12 #(maximum_duration)
        saving_dir: "results/exp/order"
    ```
5. set data:
- set data(csv) for data dir where you write for data_dir.


6. Run the structure estimation: 
- Run the OrderBase basic:
    ```sh
    cargo run order
    ```

- Run the ScoreBase experiment:
    ```sh
    cargo run exp_score
    ```

- If there is no ground truth network:
    ```sh
    cargo run est_order
    ```

## Directory Structure
### Source Code (`src`)

```
/workspace/basian_network/tree src
src
|-- common
|   |-- core
|   |   |-- mod.rs
|   |   |-- order_base.rs
|   |   |-- score_base.rs
|   |   `-- utils.rs
|   |-- est.rs
|   |-- exm.rs
|   |-- exp.rs
|   `-- mod.rs
|-- lib.rs
`-- main.rs
```

- `src/main.rs`: Entry point for the program.
- `src/lib.rs`: Entry point for the common modules.
- `src/common/`: Contains the core logic and utility functions.
  - `src/common/core/mod.rs`: Entry point for the core module.
  - `src/common/core/order_base.rs`: Implementation of the OrderBase method.
  - `src/common/core/score_base.rs`: Implementation of the ScoreBase method.
  - `src/common/core/utils.rs`: Utility functions.
  - `src/common/est.rs`: Estimation logic.
  - `src/common/exm.rs`: Experiment management logic.
  - `src/common/exp.rs`: Experiment execution logic.
  - `src/common/mod.rs`: Entry point for the common module.

### Configuration Files (`setting`)

```
/workspace/basian_network/tree setting
setting
|-- asia.dot
|-- est
|   |-- order.yml
|   `-- score.yml
|-- exm
|   |-- order.yml
|   `-- score.yml
`-- exp
    |-- order.yml
    `-- score.yml
```

- `setting/asia.dot`: Ground truth network for the Asia dataset.
- `setting/est/order.yml`: Configuration file for the OrderBase method in the estimation experiment.
- `setting/est/score.yml`: Configuration file for the ScoreBase method in the estimation experiment.
- `setting/exm/order.yml`: Configuration file for the OrderBase method in the execution experiment.
- `setting/exm/score.yml`: Configuration file for the ScoreBase method in the execution experiment.
- `setting/exp/order.yml`: Configuration file for the OrderBase method in the experiment.
- `setting/exp/score.yml`: Configuration file for the ScoreBase method in the experiment.


## Purpose

The purpose of this project is to estimate the structure of Bayesian Networks using two different methodologies:
- **OrderBase**: This method aims to reduce the execution time and finds an approximate solution.
- **ScoreBase**: This method has an execution time of O(n2^n) but finds the optimal solution.

## Configuration Files

Configuration files are used to set the parameters for each experiment. The following fields are required:

- `method`: The method to be used (`bic` or `aic` or `bdeu`).
- `bdeu_ess`: The equivalent sample size for the BDeu score.
- `data_dir`: Directory where the data files are located.
- `compare_network_dir`: Directory where the comparison network files are located.
- `data_names`: A list of data set names to be used in the experiments.
- `timeout_hours`: The timeout duration for each experiment in hours.
- `saving_dir`: Directory where the estimated BNs are saved

## Logging

The results of each experiment are logged in a timestamped text file located in the `results/exp/*` directory. Each step of the experiment, including the duration of each phase (learn, visualize, evaluate, compare), is recorded in this file.

## Example Log Entry

```text
Experiment for asia completed successfully.
learn duration: 19.414118ms
visualize duration: 158.434µs
evaluate duration: 1.554104ms
[Aic] optimized_score: -2248296.1850932767, ans_score: -2235334.5321166404
[Bic] optimized_score: -2248420.2479541353, ans_score: -2235440.871711662
[BDeu(10000.0)] optimized_score: -2312015.4954329343, ans_score: -2297897.630498507
[hamming_distance]: 6
compare duration: 7.166725ms
cpdag visualize duration: 94.747µs
```

## Ground Truth Networks

If there is a ground truth network available for a dataset, it should be stored as a `.dot` file in the `data` directory with the corresponding dataset name (e.g., `dataset1.dot`). The estimated network will also be saved in the `results` directory, following the same naming convention.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Contact

For any questions or issues, please contact [shooting3000k@gmail.com].
