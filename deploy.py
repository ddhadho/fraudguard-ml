import os
import subprocess
import boto3
import json
import time

# --- Configuration ---
AWS_REGION = os.getenv("AWS_REGION", "us-east-1") # Default region
SERVICE_NAME = os.getenv("SERVICE_NAME", "fraudguard-service")
IMAGE_TAG = os.getenv("IMAGE_TAG", "latest")
ECR_REPOSITORY_NAME = SERVICE_NAME # ECR repo name matches service name

# CloudFormation template path
CLOUDFORMATION_TEMPLATE = "deploy.yml"

def run_command(command, cwd=None, check=True):
    """Executes a shell command and returns its output."""
    print(f"Executing: {' '.join(command)}")
    result = subprocess.run(command, capture_output=True, text=True, cwd=cwd, check=check)
    if result.stdout:
        print(result.stdout)
    if result.stderr and not check: # Only print stderr if not checking for errors, as check=True will raise an exception
        print(result.stderr)
    return result.stdout.strip()

def build_and_push_docker_image():
    """Builds the Docker image and pushes it to ECR."""
    ecr_client = boto3.client("ecr", region_name=AWS_REGION)

    try:
        # Create ECR repository if it doesn't exist
        print(f"Ensuring ECR repository '{ECR_REPOSITORY_NAME}' exists...")
        ecr_client.create_repository(repositoryName=ECR_REPOSITORY_NAME)
        print(f"Repository '{ECR_REPOSITORY_NAME}' ensured.")
    except ecr_client.exceptions.RepositoryAlreadyExistsException:
        print(f"Repository '{ECR_REPOSITORY_NAME}' already exists.")
    except Exception as e:
        print(f"Error ensuring ECR repository: {e}")
        raise

    # Get ECR login password
    # This command retrieves an authentication token and logs Docker into ECR
    aws_account_id = boto3.client("sts").get_caller_identity()["Account"]
    ecr_uri = f"{aws_account_id}.dkr.ecr.{AWS_REGION}.amazonaws.com/{ECR_REPOSITORY_NAME}"
    
    print("Logging into ECR...")
    # Docker login command for ECR
    login_command = [
        "aws", "ecr", "get-login-password",
        "--region", AWS_REGION
    ]
    password = run_command(login_command)
    
    run_command([
        "docker", "login",
        "--username", "AWS",
        "--password", password,
        ecr_uri.split('/')[0] # Only the registry part
    ])
    print("Successfully logged into ECR.")

    # Build the Docker image
    full_image_name = f"{ecr_uri}:{IMAGE_TAG}"
    print(f"Building Docker image: {full_image_name}...")
    run_command([
        "docker", "build",
        "-t", full_image_name,
        "." # Build context is current directory
    ])
    print(f"Docker image '{full_image_name}' built successfully.")

    # Push the Docker image to ECR
    print(f"Pushing Docker image to ECR: {full_image_name}...")
    run_command(["docker", "push", full_image_name])
    print(f"Docker image '{full_image_name}' pushed to ECR successfully.")
    
    return full_image_name


def deploy_cloudformation_stack(image_uri):
    """Deploys or updates the CloudFormation stack."""
    cf_client = boto3.client("cloudformation", region_name=AWS_REGION)

    with open(CLOUDFORMATION_TEMPLATE, "r") as f:
        template_body = f.read()

    # Get VPC and Subnet IDs from environment variables
    vpc_id = os.getenv("VPC_ID")
    public_subnet_ids = os.getenv("PUBLIC_SUBNETS")

    if not vpc_id or not public_subnet_ids:
        print("Error: VPC_ID and PUBLIC_SUBNETS environment variables must be set for CloudFormation deployment.")
        return

    parameters = [
        {"ParameterKey": "ServiceName", "ParameterValue": SERVICE_NAME},
        {"ParameterKey": "ImageTag", "ParameterValue": IMAGE_TAG},
        {"ParameterKey": "VpcId", "ParameterValue": vpc_id},
        {"ParameterKey": "PublicSubnetIds", "ParameterValue": public_subnet_ids},
    ]

    stack_name = f"{SERVICE_NAME}-ecs-stack"
    print(f"Deploying CloudFormation stack '{stack_name}'...")

    try:
        cf_client.describe_stacks(StackName=stack_name)
        # Stack exists, update it
        print(f"Stack '{stack_name}' exists, attempting to update...")
        cf_client.update_stack(
            StackName=stack_name,
            TemplateBody=template_body,
            Parameters=parameters,
            Capabilities=["CAPABILITY_IAM"] # Required for IAM roles
        )
        waiter = cf_client.get_waiter("stack_update_complete")
        print("Waiting for stack update to complete...")
        waiter.wait(StackName=stack_name)
        print(f"Stack '{stack_name}' updated successfully.")
    except cf_client.exceptions.ClientError as e:
        if "does not exist" in str(e):
            # Stack does not exist, create it
            print(f"Stack '{stack_name}' does not exist, attempting to create...")
            cf_client.create_stack(
                StackName=stack_name,
                TemplateBody=template_body,
                Parameters=parameters,
                Capabilities=["CAPABILITY_IAM"]
            )
            waiter = cf_client.get_waiter("stack_create_complete")
            print("Waiting for stack creation to complete...")
            waiter.wait(StackName=stack_name)
            print(f"Stack '{stack_name}' created successfully.")
        else:
            raise # Re-raise other ClientErrors

def main():
    print("--- Starting Deployment Process ---")

    # 1. Build and push Docker image to ECR
    try:
        image_uri = build_and_push_docker_image()
        print(f"Image URI: {image_uri}")
    except Exception as e:
        print(f"Docker build and push failed: {e}")
        exit(1)

    # 2. Deploy/Update CloudFormation stack
    try:
        deploy_cloudformation_stack(image_uri)
    except Exception as e:
        print(f"CloudFormation deployment failed: {e}")
        exit(1)

    print("--- Deployment Process Finished ---")

if __name__ == "__main__":
    main()
