
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
    - `setting/setting_oreder.yml` for one dataset
    - `setting/setting_score.yml` for one dataset
    - `exp/order.yml` for many datasets
    - `exp/score.yml` for many datasets
    - `est/setting_order.yml` for one dataset whose you have no answer
    - `est/setting_score.yml` for one dataset whose you have no answer

    Example configuration file (`setting/setting_oreder.yml`):
    ```yaml
    method: "bic" #(aic, bic, bdeu)
    bdeu_ess: 10.0 #(for bdeu)
    data_dir: "./data" #(data dir)
    compare_network_dir: "./data" #(ans dot file dir)
    data_names:
      - "dataset1" #(dataset name ({dataset}.csv))
      - "dataset2"
    timeout_hours: 12 #(maximum_duration)
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
    ```



5. Run the structure estimation: 
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

## Project Structure

- `src/`
  - `main.rs`: Entry point for the program.
  - `order_base.rs`: Implementation of the OrderBase method.
  - `score_base.rs`: Implementation of the ScoreBase method.
  - `exp.rs`: Experiment management and execution logic.

- `exp/`
  - `order.yml`: Configuration file for the OrderBase experiment.
  - `score.yml`: Configuration file for the ScoreBase experiment.
  - `result_order_{timestamp}.txt`: Results of the OrderBase experiment.
  - `result_score_{timestamp}.txt`: Results of the ScoreBase experiment.

- `data/`
  - `dataset1.csv`: Data of dataset1
  - `dataset1.dot`: Ground truth network for dataset1.
  - `dataset2.csv`: Data of dataset2
  - `dataset2.dot`: Ground truth network for dataset2.

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

## Logging

The results of each experiment are logged in a timestamped text file located in the `exp/` directory. Each step of the experiment, including the duration of each phase (learn, visualize, evaluate, compare), is recorded in this file.

## Example Log Entry

```text
Experiment for asia completed successfully.
analyze duration: 1.756882901s
visualize duration: 178.487µs
evaluate duration: 2.984494ms
[Aic] optimized_score: -2235334.53211664, ans_score: -2235334.53211664
[Bic] optimized_score: -2235440.8717116616, ans_score: -2235440.8717116616
[BDeu(10000.0)] optimized_score: -2297897.6304985075, ans_score: -2297897.6304985075
[hamming_distance]: 0
compare duration: 48.985137ms
```

## Ground Truth Networks

If there is a ground truth network available for a dataset, it should be stored as a `.dot` file in the `data` directory with the corresponding dataset name (e.g., `dataset1.dot`). The estimated network will also be saved in the `figure` directory, following the same naming convention.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request on GitHub.

## Contact

For any questions or issues, please contact [your-email@example.com].
