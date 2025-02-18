#include "my_bundle_adjustment.h"

int main(int argc, char **argv)
{
    google::InitGoogleLogging(argv[0]);
    // 读取图片
    cv::Mat img1 = cv::imread("../img/1.png", cv::IMREAD_COLOR);
    cv::Mat img2 = cv::imread("../img/2.png", cv::IMREAD_COLOR);
    cv::Mat img1_depth = cv::imread("../img/1_depth.png", cv::IMREAD_UNCHANGED);

    // 提取ORB特征点
    std::vector<cv::KeyPoint> KeyPoints1, KeyPoints2;
    std::vector<cv::DMatch> matches;
    find_feature_matches(img1, img2, KeyPoints1, KeyPoints2, matches);
    LOG(ERROR) << "找到" << matches.size() << "对匹配点!";
    
    std::vector<cv::Point2f> points2_pixel; // 第二张图中特征点的像素坐标
    std::vector<cv::Point3f> points1_camera; // 第一张图中特征点的相机坐标

    cv::Mat K = (cv::Mat_<double>(3, 3) << 520.9, 0, 325.1, 0, 521.0, 249.7, 0, 0, 1); // 相机内参

    for (auto &match : matches)
    {
        cv::Point2f point2_pixel = KeyPoints2[match.trainIdx].pt;
        ushort d = img1_depth.ptr<ushort>(int(KeyPoints1[match.queryIdx].pt.y))[int(KeyPoints1[match.queryIdx].pt.x)];
        if (d == 0)
            continue;
        double dd = d / 5000.0;
        cv::Point3f point1_camera = pixel2Camera(KeyPoints1[match.queryIdx].pt, K, dd);
        points2_pixel.push_back(point2_pixel);
        points1_camera.push_back(point1_camera);
    }

    LOG(ERROR) << "3d-2d pairs: " << points1_camera.size();

    cv::Mat r, t, R; // r是用旋转向量描述的旋转，可以用罗德里格斯公式转化为旋转矩阵
    cv::solvePnP(points1_camera, points2_pixel, K,
                 cv::Mat(), r, t, false, cv::SOLVEPNP_EPNP);
    cv::Rodrigues(r, R);
    LOG(ERROR) << "R = " << R << "\n t = " << t;

    // 把3D点和2D点转换成Eigen格式
    Vec3dPoints points_3d_eigen;
    Vec2dPoints points_2d_eigen;
    for (int i = 0; i < points1_camera.size(); i++)
    {
        Eigen::Vector3d point3d(points1_camera[i].x, points1_camera[i].y, points1_camera[i].z);
        points_3d_eigen.push_back(point3d);
        Eigen::Vector2d point2d(points2_pixel[i].x, points2_pixel[i].y);
        points_2d_eigen.push_back(point2d);
    }

    Eigen::Matrix3d R_eigen;
    R_eigen << R.at<double>(0, 0), R.at<double>(0, 1), R.at<double>(0, 2),
               R.at<double>(1, 0), R.at<double>(1, 1), R.at<double>(1, 2),
               R.at<double>(2, 0), R.at<double>(2, 1), R.at<double>(2, 2);
    Sophus::SE3d camera_pose_init;
    Sophus::SE3d camera_pose_optimized;
    Vec3dPoints points_3d_optimized;
    camera_pose_init.setRotationMatrix(R_eigen);
    camera_pose_init.translation() = Eigen::Vector3d(t.at<double>(0, 0), t.at<double>(1, 0), t.at<double>(2, 0));
    LOG(ERROR) << "camera_pose_init = " << camera_pose_init.matrix();
    CameraPoseGaussNewton(points_3d_eigen, points_2d_eigen, K, camera_pose_init, camera_pose_optimized, 10);
    LOG(ERROR) << "使用高斯牛顿法优化后的相机位姿为: " << camera_pose_optimized.matrix();

    GaussNewton(points_3d_eigen, points_2d_eigen, K, camera_pose_init, camera_pose_optimized, points_3d_optimized, 100);
    LOG(ERROR) << "使用高斯牛顿法同时优化3D点和相机位姿: " 
               << "\n" << "T = " << camera_pose_optimized.matrix();
}
