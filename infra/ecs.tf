resource "aws_ecs_cluster" "main" {
  name = var.project

  setting {
    name  = "containerInsights"
    value = "disabled"
  }
}

resource "aws_cloudwatch_log_group" "api" {
  name              = "/ecs/${var.project}"
  retention_in_days = 14
}

# ------------------------------------------------------------------ IAM

data "aws_iam_policy_document" "ecs_tasks_assume" {
  statement {
    actions = ["sts:AssumeRole"]

    principals {
      type        = "Service"
      identifiers = ["ecs-tasks.amazonaws.com"]
    }
  }
}

# Used by ECS itself to pull the image, write logs and inject the secrets. The app makes no AWS
# calls, so the task has no task role at all.
resource "aws_iam_role" "execution" {
  name               = "${var.project}-execution"
  assume_role_policy = data.aws_iam_policy_document.ecs_tasks_assume.json
}

resource "aws_iam_role_policy_attachment" "execution" {
  role       = aws_iam_role.execution.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AmazonECSTaskExecutionRolePolicy"
}

data "aws_iam_policy_document" "read_secrets" {
  statement {
    actions   = ["ssm:GetParameters"]
    resources = values(local.container_secrets)
  }
}

resource "aws_iam_role_policy" "read_secrets" {
  name   = "read-secrets"
  role   = aws_iam_role.execution.id
  policy = data.aws_iam_policy_document.read_secrets.json
}

# ------------------------------------------------------------- task + service

# ONNX_THREADS: ONNX Runtime would otherwise start one thread per host core, far more than the
# share of a CPU Fargate gives the task.
locals {
  container_env = {
    LLM_PROVIDER    = "anthropic"
    LLM_MODEL       = var.llm_model
    QDRANT_URL      = var.qdrant_url
    COLLECTION_NAME = var.collection_name
    ONNX_THREADS    = var.cpu <= 512 ? "1" : "2"
    LOG_LEVEL       = "INFO"
  }
}

resource "aws_ecs_task_definition" "api" {
  family                   = var.project
  requires_compatibilities = ["FARGATE"]
  network_mode             = "awsvpc"
  cpu                      = var.cpu
  memory                   = var.memory
  execution_role_arn       = aws_iam_role.execution.arn

  runtime_platform {
    operating_system_family = "LINUX"
    cpu_architecture        = "X86_64"
  }

  container_definitions = jsonencode([{
    name      = "api"
    image     = "${aws_ecr_repository.api.repository_url}:${var.image_tag}"
    essential = true

    portMappings = [{ containerPort = 8000, protocol = "tcp" }]

    environment = [for k, v in local.container_env : { name = k, value = v }]
    secrets     = [for k, arn in local.container_secrets : { name = k, valueFrom = arn }]

    logConfiguration = {
      logDriver = "awslogs"
      options = {
        awslogs-group         = aws_cloudwatch_log_group.api.name
        awslogs-region        = var.aws_region
        awslogs-stream-prefix = "api"
      }
    }
  }])
}

resource "aws_ecs_service" "api" {
  name            = "api"
  cluster         = aws_ecs_cluster.main.id
  task_definition = aws_ecs_task_definition.api.arn
  desired_count   = var.desired_count
  launch_type     = "FARGATE"

  # First start downloads the embedding, BM25 and reranker models (~1-2 min).
  health_check_grace_period_seconds = 300

  # A deployment that never gets healthy is stopped and rolled back automatically.
  deployment_circuit_breaker {
    enable   = true
    rollback = true
  }

  network_configuration {
    subnets          = aws_subnet.public[*].id
    security_groups  = [aws_security_group.task.id]
    assign_public_ip = true
  }

  load_balancer {
    target_group_arn = aws_lb_target_group.api.arn
    container_name   = "api"
    container_port   = 8000
  }

  depends_on = [aws_lb_listener_rule.from_cloudfront]
}
