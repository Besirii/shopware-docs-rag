# Build the API image and push it to the ECR repository Terraform created.
# Needed once before the first full `terraform apply`; afterwards GitHub Actions does it on every push.
#   .\infra\push-image.ps1
param(
    [string]$Region = "eu-central-1",
    [string]$Repository = "shopware-docs-rag"
)
$ErrorActionPreference = "Stop"

$repo = aws ecr describe-repositories --region $Region --repository-names $Repository --query "repositories[0].repositoryUri" --output text
if ($LASTEXITCODE -ne 0 -or -not $repo) { throw "No ECR repository yet - run: terraform apply -target=aws_ecr_repository.api" }
$registry = $repo.Split("/")[0]

aws ecr get-login-password --region $Region | docker login --username AWS --password-stdin $registry
if ($LASTEXITCODE -ne 0) { throw "docker login to ECR failed - check 'aws configure' and the region" }

docker build -t "${repo}:latest" "$PSScriptRoot\.."
if ($LASTEXITCODE -ne 0) { throw "docker build failed" }

docker push "${repo}:latest"
if ($LASTEXITCODE -ne 0) { throw "docker push failed" }
"Pushed ${repo}:latest"
