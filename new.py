import pygame
import sys
import random
import numpy as np
from sklearn.tree import DecisionTreeClassifier

# Initialize Pygame
pygame.init()

# Constants
WIDTH, HEIGHT = 600, 400
WHITE = (255, 255, 255)
FPS = 60
PADDLE_WIDTH, PADDLE_HEIGHT = 80, 10
BALL_SIZE = 15
BRICK_WIDTH, BRICK_HEIGHT = 60, 20
NUM_BRICKS = 40

# Create the game window
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Brick Breaker")

# Paddle settings
paddle = pygame.Rect(WIDTH // 2 - PADDLE_WIDTH // 2, HEIGHT - 20, PADDLE_WIDTH, PADDLE_HEIGHT)
paddle_speed = 8

# Ball settings
ball = pygame.Rect(WIDTH // 2 - BALL_SIZE // 2, HEIGHT // 2 - BALL_SIZE // 2, BALL_SIZE, BALL_SIZE)
ball_speed_x, ball_speed_y = 5, 5

# Bricks settings
bricks = [pygame.Rect(i * BRICK_WIDTH + 50, j * BRICK_HEIGHT + 50, BRICK_WIDTH, BRICK_HEIGHT)
          for i in range(8) for j in range(5)]

# Scoring
score = 0

# Intelligent Agent
class IntelligentAgent:
    def __init__(self, paddle):
        self.paddle = paddle
        self.decision_tree = DecisionTreeClassifier()

    def table_driven_move(self, percept):
        # Simple heuristic: Move paddle towards the ball's x-coordinate
        ball_x = percept['ball']['x']
        paddle_x = percept['paddle']['x']

        if ball_x < paddle_x and paddle.left > 0:
            return 'LEFT'
        elif ball_x > paddle_x and paddle.right < WIDTH:
            return 'RIGHT'
        else:
            return 'NONE'  # No movement

    def rule_based_move(self, percept):
        # Rule-based heuristic: Move paddle based on ball position and velocity
        ball_x = percept['ball']['x']
        ball_speed_x = percept['ball']['speed_x']

        if ball_speed_x > 0 and ball_x > WIDTH // 2 and paddle.right < WIDTH:
            return 'RIGHT'
        elif ball_speed_x < 0 and ball_x < WIDTH // 2 and paddle.left > 0:
            return 'LEFT'
        else:
            return 'NONE'

    def ml_based_move(self, percept):
        # ML-based heuristic using a simple decision tree
        data = np.array([[percept['ball']['x'], percept['ball']['speed_x']]])
        move = self.decision_tree.predict(data)[0]
        return move

# Create the intelligent agent
agent = IntelligentAgent(paddle)

# Function to update the game state
def update(percept, action):
    if action == 'LEFT' and paddle.left > 0:
        paddle.x -= paddle_speed
    elif action == 'RIGHT' and paddle.right < WIDTH:
        paddle.x += paddle_speed

# Function to draw the paddle
def draw_paddle(paddle):
    pygame.draw.rect(screen, WHITE, paddle)

# Function to draw the ball
def draw_ball(ball):
    pygame.draw.ellipse(screen, WHITE, ball)

# Function to draw the bricks
def draw_bricks(bricks):
    for brick in bricks:
        pygame.draw.rect(screen, WHITE, brick)

# Function to draw the score
def draw_score(score):
    font = pygame.font.Font(None, 36)
    score_text = font.render(f"Score: {score}", True, WHITE)
    screen.blit(score_text, (10, 10))

# Function to handle collisions with bricks
def handle_brick_collision(ball, bricks, score, ball_speed_y):
    for brick in bricks:
        if ball.colliderect(brick):
            bricks.remove(brick)
            score += 10
            ball_speed_y = -ball_speed_y
            break
    return score, ball_speed_y

# Function to train the decision tree with simple data
def train_decision_tree(agent):
    # Simple training data: ball_x, ball_speed_x, move
    X = np.array([[150, 5], [400, -5], [200, 5], [300, -3]])
    y = np.array(['RIGHT', 'LEFT', 'RIGHT', 'LEFT'])
    agent.decision_tree.fit(X, y)

def end_game():
    pygame.quit()
    sys.exit()

# Performance Metrics
class PerformanceMetrics:
    def __init__(self):
        self.decision_accuracy = 0
        self.total_decisions = 0

    def update_decision_accuracy(self, expected_action, actual_action):
        self.total_decisions += 1
        if expected_action == actual_action:
            self.decision_accuracy += 1

    def get_metrics(self):
        accuracy_percentage = (self.decision_accuracy / self.total_decisions) * 100
        return {'Decision Accuracy (%)': accuracy_percentage}

# Initialize performance metrics
performance_metrics = PerformanceMetrics()

# Main game loop
clock = pygame.time.Clock()

# Train the decision tree
train_decision_tree(agent)

while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            end_game()

    if score >= 400:
        end_game()

    # Move the ball
    ball.x += ball_speed_x
    ball.y += ball_speed_y

    # Ball collisions with walls
    if ball.top <= 0:
        ball_speed_y = -ball_speed_y
    elif ball.right >= WIDTH or ball.left <= 0:
        ball_speed_x = -ball_speed_x

    # Ball collision with paddle
    if ball.colliderect(paddle) and ball_speed_y > 0:
        ball_speed_y = -ball_speed_y

    # Handle collisions with bricks
    score, ball_speed_y = handle_brick_collision(ball, bricks, score, ball_speed_y)

    # Clear the screen
    screen.fill((0, 0, 0))

    # Draw game elements
    draw_paddle(paddle)
    draw_ball(ball)
    draw_bricks(bricks)
    draw_score(score)

    # Get percepts
    percept = {
        'paddle': {'x': paddle.x},
        'ball': {'x': ball.x, 'y': ball.y, 'speed_x': ball_speed_x}
    }

    # Use table-driven approach to get action
    action_table = agent.table_driven_move(percept)

    # Update decision accuracy in performance metrics
    performance_metrics.update_decision_accuracy(action_table, action_table)

    # Update game state based on table-driven action
    update(percept, action_table)

    # Update the display
    pygame.display.flip()

    # Cap the frame rate
    clock.tick(FPS)

    # Display performance metrics at the end of the game
    if score >= 400:
        metrics = performance_metrics.get_metrics()
        print("\nPerformance Metrics:")
        for key, value in metrics.items():
            print(f"{key}: {value}")
