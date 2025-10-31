import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from geometry_msgs.msg import Twist

class VehicleController(Node):
    def __init__(self):
        super().__init__('vehicle_controller')
        self.subscription = self.create_subscription(
            String,
            '/atc/orders',
            self.listener_callback,
            10
        )
        self.publisher = self.create_publisher(Twist, '/cmd_vel', 10)
        self.get_logger().info("VehicleController Node Started")

    def listener_callback(self, msg):
        self.get_logger().info(f"\nReceived Command: {msg.data}")
        self.execute_command(msg.data)

    def execute_command(self, command):
        twist = Twist()
        if command.lower() == "stop":
            self.get_logger().info("Stopping vehicle...")
            twist.linear.x = 0.0
            twist.angular.z = 0.0
        elif command.lower() == "advance":
            self.get_logger().info("Advancing...")
            twist.linear.x = 1.0
            twist.angular.z = 0.0
        elif command.lower() == "turn_left":
            self.get_logger().info("Turning left...")
            twist.linear.x = 0.5
            twist.angular.z = 1.0
        elif command.lower() == "turn_right":
            self.get_logger().info("Turning right...")
            twist.linear.x = 0.5
            twist.angular.z = -1.0
        else:
            self.get_logger().info("Unknown command. Stopping vehicle.")
            twist.linear.x = 0.0
            twist.angular.z = 0.0

        self.publisher.publish(twist)
        self.get_logger().info(f"Published Twist: {twist}\n")


def main(args=None):
    rclpy.init(args=args)
    node = VehicleController()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
