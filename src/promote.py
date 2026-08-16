from mlflow import MlflowClient

MODEL_NAME = "forge-ml-classifier"

def main():
    client = MlflowClient()
    versions = client.search_model_versions(f"name='{MODEL_NAME}'")
    latest_version = max(versions, key=lambda version: int(version.version))
    client.set_registered_model_alias(name=MODEL_NAME, 
                                      alias="champion", 
                                      version=latest_version.version)
    print(f"Promoted {MODEL_NAME} "
          f"Version {latest_version.version} "
          f"to champion ")

if __name__ == "__main__":
    main()

