# Architecture
Right now the architecture is pretty simple.
Run multiple runs within the experiment with shift in values(parameters)
and log the results along with logging other params and metrics

                Python process
                      │
                      │ mlflow.start_run()
                      ▼
                 MLflow Tracking
                 ┌────────────┐
                 │ experiment │
                 └─────┬──────┘
                       │
                 ┌─────▼─────┐
                 │    run    │
                 └─────┬─────┘
             ┌─────────┼─────────┐
             ▼         ▼         ▼
         params     metrics    model
