#ifndef MY_BUNDLE_ADJUSTMENT_H
#define MY_BUNDLE_ADJUSTMENT_H

#include <opencv4/opencv2/opencv.hpp>
#include <opencv4/opencv2/core/core.hpp>
#include <opencv4/opencv2/features2d/features2d.hpp>
#include <eigen3/Eigen/Dense>
#include <sophus/se3.hpp>
#include <glog/logging.h>

using Vec3dPoints = std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>>;
using Vec2dPoints = std::vector<Eigen::Vector2d, Eigen::aligned_allocator<Eigen::Vector2d>>;

void find_feature_matches(
    const cv::Mat &img1, const cv::Mat &img2,
    std::vector<cv::KeyPoint> &keypoints1,
    std::vector<cv::KeyPoint> &keypoints2,
    std::vector<cv::DMatch> &matches);

cv::Point3f pixel2Camera(cv::Point2f &point_pixel, cv::Mat &K, const float &depth);

/**
 * @brief 现在考虑这个数学问题:已知第一帧图像中的许多个特征点的3d点坐标，和这些特征点在第二帧
 * 图像中的2d像素坐标，以及相机内参，相机位姿的初始值(由pnp给出一个较好的初值)，采用
 * 最小二乘法构建误差项，总的误差为每个匹配点的误差之和(所以总的H矩阵和g向量也是累加)
 * @param points_3d 一个向量，存储第一帧图像中特征点的相机坐标
 * @param points_2d 一个向量，存储第二帧图像中特征点的像素坐标
 * @param K 相机内参
 * @param camera_pose_init 相机初始位姿(外参)，实际上是第二帧图像相对于第一帧图像的位姿
 * @param camera_pose_optimized 优化后的相机位姿
 * @param max_iterations 最大迭代次数
 */
void CameraPoseGaussNewton(const Vec3dPoints &points_3d,
                           const Vec2dPoints &points_2d,
                           const cv::Mat &K,
                           const Sophus::SE3d &camera_pose_init,
                           Sophus::SE3d &camera_pose_optimized,
                           const int &max_iterations);

/**
 * @brief 3D-2D bundle adjustment using Gauss-Newton method
 * @param points_3d 存储第一帧图像中特征点的相机坐标
 * @param points_2d 存储第二帧图像中特征点的像素坐标
 * @param K 相机内参
 * @param camera_pose_init 相机位姿的初始值
 * @param camera_pose_optimized 优化后的相机位姿
 * @param points_3d_optimized 优化后的3D点坐标
 * @param max_iterations 最大迭代次数
 */
void GaussNewton(const Vec3dPoints &points_3d,
                 const Vec2dPoints &points_2d,
                 const cv::Mat &K,
                 const Sophus::SE3d &camera_pose_init,
                 Sophus::SE3d &camera_pose_optimized,
                 Vec3dPoints &points_3d_optimized,
                 const int &max_iterations);
#endif