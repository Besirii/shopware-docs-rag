# Application Load Balancer in front of the Fargate task. Reachable only from CloudFront (security
# group + secret header); the default action answers 403 to anything without the header.

resource "aws_lb" "api" {
  name               = var.project
  load_balancer_type = "application"
  internal           = false
  security_groups    = [aws_security_group.alb.id]
  subnets            = aws_subnet.public[*].id

  # The agent can take ~15 s; keep connections open well beyond that.
  idle_timeout = 120
}

resource "aws_lb_target_group" "api" {
  name                 = "${var.project}-api"
  port                 = 8000
  protocol             = "HTTP"
  target_type          = "ip"
  vpc_id               = aws_vpc.main.id
  deregistration_delay = 30

  # /docs answers 200 as soon as the app is up. /health is not used here: it returns 503 when the
  # LLM key is wrong, and that should not make ECS restart a healthy container in a loop.
  health_check {
    path                = "/docs"
    matcher             = "200"
    interval            = 15
    timeout             = 5
    healthy_threshold   = 2
    unhealthy_threshold = 5
  }
}

resource "aws_lb_listener" "http" {
  load_balancer_arn = aws_lb.api.arn
  port              = 80
  protocol          = "HTTP"

  default_action {
    type = "fixed-response"

    fixed_response {
      content_type = "text/plain"
      message_body = "Forbidden"
      status_code  = "403"
    }
  }
}

resource "aws_lb_listener_rule" "from_cloudfront" {
  listener_arn = aws_lb_listener.http.arn
  priority     = 10

  condition {
    http_header {
      http_header_name = "X-Origin-Verify"
      values           = [random_password.origin_verify.result]
    }
  }

  action {
    type             = "forward"
    target_group_arn = aws_lb_target_group.api.arn
  }
}
