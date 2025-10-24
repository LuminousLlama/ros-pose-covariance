import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseWithCovarianceStamped
import numpy as np


def main():
    rclpy.init()
    node = Node("playground")

    pub = node.create_publisher(PoseWithCovarianceStamped, "pose", 10)

    rate = node.create_rate(50)
    while rclpy.ok():
        rclpy.spin_once(node=node, timeout_sec=0.05)

        msg = PoseWithCovarianceStamped()
        msg.header.stamp = node.get_clock().now().to_msg()
        msg.header.frame_id = "map"

        cov_mat = np.zeros((6, 6))

        # for fun rotate angle with respect to time. 1 rot every 10 seconds
        time_in_seconds = node.get_clock().now().nanoseconds * 1e-9
        angle = (time_in_seconds % 10.0) * (2 * np.pi / 10.0)
        rot_matrix = np.array(
            [
                [np.cos(angle), -np.sin(angle)],
                [np.sin(angle), np.cos(angle)],
            ]
        )

        cov_circle = np.array(
            [
                [1, 0],
                [0, 1],
            ]
        )

        # variance is standard deviation squared
        var_major = 1.0**2
        var_minor = 0.5**2
        scale_axis = np.array(
            [
                [var_major, 0],
                [0, var_minor],
            ]
        )

        # remember matrix multiplication is right to left

        # rot_matrix.T is same as rot_matrix inverse because it is a orthonormal matrix
        # this makes sense because it does not change lengths of vectors and keeps vectors perpendicular
        # we use rot_matrix.T instead of inverse b/c is it faster to compute and the standard convention

        # 1. rot_matrix.T puts us in the rotated frame
        # 2. cov_circle is just a identity matrix so nothing happens but i put it here for conceptual understanding of the covariance matrix entering the rotated frame
        # 3. scale_axis is scaling the cov circle to match the shape of a ellipse
        # 4. rot_matrix puts us back into the world frame (you can think rot_matrix and rot_matrix.T as canceling each other out)

        cov_xy = rot_matrix @ scale_axis @ cov_circle @ rot_matrix.T

        eigen_values = np.linalg.eigvals(cov_xy)
        print("eigen values: ", eigen_values)

        cov_mat[0, 0] = cov_xy[0, 0]
        cov_mat[1, 1] = cov_xy[1, 1]
        cov_mat[0, 1] = cov_xy[0, 1]
        cov_mat[1, 0] = cov_xy[1, 0]

        msg.pose.covariance = cov_mat.flatten().tolist()

        pub.publish(msg)

        rate.sleep()

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
