#include "my_bundle_adjustment.h"

void find_feature_matches(
    const cv::Mat &img1, const cv::Mat &img2,
    std::vector<cv::KeyPoint> &keypoints1,
    std::vector<cv::KeyPoint> &keypoints2,
    std::vector<cv::DMatch> &matches)
{
    // 初始化
    cv::Mat descriptors1, descriptors2;
    cv::Ptr<cv::FeatureDetector> detector = cv::ORB::create();
    cv::Ptr<cv::DescriptorExtractor> descriptor = cv::ORB::create();
    cv::BFMatcher matcher(cv::NORM_HAMMING);
    // 第一步:检测Oriented FAST角点位置
    detector->detect(img1, keypoints1);
    detector->detect(img2, keypoints2);

    // 第二步:根据角点位置计算BRIEF描述子
    descriptor->compute(img1, keypoints1, descriptors1);
    descriptor->compute(img2, keypoints2, descriptors2);

    // 第三步:对两幅图像中的BRIEF描述子进行匹配,使用Hamming距离
    std::vector<cv::DMatch> matches_all;
    matcher.match(descriptors1, descriptors2, matches_all);
    // 第四步：匹配点对筛选
    double min_dist = 10000, max_dist = 0;

    // 找出所有匹配之间的最小距离和最大距离,即是最相似的和最不相似的两组点之间的距离
    for (int i = 0; i < descriptors1.rows; i++)
    {
        double dist = matches_all[i].distance;
        if (dist < min_dist)
            min_dist = dist;
        if (dist > max_dist)
            max_dist = dist;
    }

    // 当描述子之间的距离大于两倍的最小距离时,即认为匹配有误
    for (int i = 0; i < descriptors1.rows; i++)
    {
        if (matches_all[i].distance <= std::max(2 * min_dist, 30.0))
        {
            matches.push_back(matches_all[i]);
        }
    }
}

cv::Point3f pixel2Camera(cv::Point2f &point_pixel, cv::Mat &K, const float &depth)
{
    float x_c = (point_pixel.x - K.at<double>(0, 2)) * depth / K.at<double>(0, 0);
    float y_c = (point_pixel.y - K.at<double>(1, 2)) * depth / K.at<double>(1, 1);
    float z_c = depth;
    return cv::Point3f(x_c, y_c, z_c);
}


void CameraPoseGaussNewton(const Vec3dPoints &points_3d,
                           const Vec2dPoints &points_2d,
                           const cv::Mat &K,
                           const Sophus::SE3d &camera_pose_init,
                           Sophus::SE3d &camera_pose_optimized,
                           const int &max_iterations)
{
    double fx = K.at<double>(0, 0);
    double fy = K.at<double>(1, 1);
    double cx = K.at<double>(0, 2);
    double cy = K.at<double>(1, 2);
    camera_pose_optimized = camera_pose_init;
    double last_error = 999999999; // 上一次迭代的误差
    bool converged = false; // 是否收敛

    // 迭代优化相机位姿
    for (int i = 0; i < max_iterations; i++)
    {
        // 总的误差项
        double total_error = 0;
        // 总的H矩阵
        Eigen::Matrix<double, 6, 6> H = Eigen::Matrix<double, 6, 6>::Zero();
        // 总的g向量
        Eigen::Matrix<double, 6, 1> g = Eigen::Matrix<double, 6, 1>::Zero();
        for (int j = 0; j < points_3d.size(); j++)
        {
            // 把第一帧图像中相机坐标系下的3d点转换到第二帧图像的相机坐标系下
            Eigen::Vector3d point3d_camera = camera_pose_optimized * points_3d[j];
            // 再把上述相机坐标系下的3d点转换到像素坐标系下，得到预测的2d点
            Eigen::Vector2d point_pixel = Eigen::Vector2d(fx * point3d_camera[0] / point3d_camera[2] + cx,
                                                          fy * point3d_camera[1] / point3d_camera[2] + cy);

            double X = point3d_camera[0];
            double Y = point3d_camera[1];                                              
            double Z_inv = 1.0 / point3d_camera[2];
            double Z_inv_2 = Z_inv * Z_inv;
            Eigen::Vector2d error = points_2d[j] - point_pixel;
            total_error += error.squaredNorm();
            // 计算Jacobian矩阵
            Eigen::Matrix<double, 2, 6> Jacobian;
            Jacobian << -fx * Z_inv, 0, fx * X * Z_inv_2,
                        fx * X * Y * Z_inv_2, -fx - fx * X * X * Z_inv_2,
                        fx * Y * Z_inv, 0, -fy * Z_inv,
                        fy * Y * Z_inv_2, fy + fy * Y * Y * Z_inv_2,
                        -fy * X * Y * Z_inv_2, -fy * X * Z_inv;
            // 计算H矩阵和g向量
            H += Jacobian.transpose() * Jacobian;
            g += -Jacobian.transpose() * error;
        }
        // 求解增量方程 H * delta_x = g
        Eigen::Matrix<double, 6, 1> delta_x = H.ldlt().solve(g);
        if (std::isnan(delta_x[0]))
        {
            LOG(WARNING) << "result is nan!";
            break;
        }

        if (i > 0 && total_error > last_error)
        {
            LOG(WARNING) << "total error increased: " << total_error << ", " << last_error;
            break;
        }

        last_error = total_error;
        // 更新相机位姿
        Sophus::SE3d delta_pose = Sophus::SE3d::exp(delta_x);
        camera_pose_optimized = delta_pose * camera_pose_optimized;
        LOG(ERROR) << "iteration " << i << " total error: " << total_error;
        if (delta_x.norm() < 1e-6)
        {
            LOG(INFO) << "converged after " << i << " iterations";
            converged = true;
            break;
        }
    }
    if (!converged)
    {
        LOG(WARNING) << "not converged";
    }
}

void GaussNewton(const Vec3dPoints &points_3d,
                 const Vec2dPoints &points_2d,
                 const cv::Mat &K,
                 const Sophus::SE3d &camera_pose_init,
                 Sophus::SE3d &camera_pose_optimized,
                 Vec3dPoints &points_3d_optimized,
                 const int &max_iterations)
{
    double fx = K.at<double>(0, 0);
    double fy = K.at<double>(1, 1);
    double cx = K.at<double>(0, 2);
    double cy = K.at<double>(1, 2);
    camera_pose_optimized = camera_pose_init;
    points_3d_optimized = points_3d;
    double last_error = 999999999;
    bool converged = false;

    for(int iter = 0; iter < max_iterations; iter++)
    {
        // 总的误差项
        double total_error = 0;

        // 匹配特征点的数量
        const int n = points_3d.size();

        // 误差向量
        Eigen::VectorXd error_vector = Eigen::VectorXd::Zero(2 * n);

        // 总的雅可比矩阵
        Eigen::MatrixXd Jacobian = Eigen::MatrixXd::Zero(2 * n, 6 + 3 * n);

        // 总的H矩阵
        Eigen::MatrixXd H = Eigen::MatrixXd::Zero(6 + 3 * n, 6 + 3 * n);

        // 总的g向量
        Eigen::VectorXd g = Eigen::VectorXd::Zero(6 + 3 * n);
        for(int i = 0; i < points_3d.size(); i++)
        {
            // 把第一帧图像中相机坐标系下的3d点转换到第二帧图像的相机坐标系下
            Eigen::Vector3d point3d_camera = camera_pose_optimized * points_3d_optimized[i];
            // 再把上述相机坐标系下的3d点转换到像素坐标系下，得到预测的2d点
            Eigen::Vector2d point_pixel = Eigen::Vector2d(fx * point3d_camera[0] / point3d_camera[2] + cx,
                                                          fy * point3d_camera[1] / point3d_camera[2] + cy);
            double X = point3d_camera[0];
            double Y = point3d_camera[1];                                              
            double Z_inv = 1.0 / point3d_camera[2];
            double Z_inv_2 = Z_inv * Z_inv;
            Eigen::Vector2d error = points_2d[i] - point_pixel;

            // 向误差向量中填充数据
            error_vector[2 * i] = error[0]; // 第i个特征点的x方向的误差
            error_vector[2 * i + 1] = error[1]; // 第i个特征点的y方向的误差
            // 向总的雅可比矩阵中填充数据
            Jacobian.block<2, 6>(2 * i, 0) << -fx * Z_inv, 0, 
                                              fx * X * Z_inv_2,
                                              fx * X * Y * Z_inv_2,
                                              -fx - fx * X * X * Z_inv_2,
                                              fx * Y * Z_inv, 0,
                                              -fy * Z_inv,
                                              fy * Y * Z_inv_2,
                                              fy + fy * Y * Y * Z_inv_2,
                                              -fy * X * Y * Z_inv_2,
                                              -fy * X * Z_inv;
            Eigen::Matrix<double, 2, 3> tmp1;
            Sophus::SO3d R = camera_pose_optimized.so3();
            tmp1 << fx * Z_inv, 0, -fx * X * Z_inv_2,
                    0, fy * Z_inv, -fy * Y * Z_inv_2;

            Jacobian.block<2, 3>(2 * i, 6 + 3 * i) = -tmp1 * R.matrix();
        }
        // 计算H矩阵和g向量
        H = Jacobian.transpose() * Jacobian; // 维度为(6 + 3 * n) * (6 + 3 * n)
        g = -Jacobian.transpose() * error_vector; // 维度为(6 + 3 * n) * 1

        // 稀疏矩阵求解增量方程 H * delta_x = g
        Eigen::VectorXd delta_x = H.ldlt().solve(g);

        if (std::isnan(delta_x[0]))
        {
            LOG(WARNING) << "result is nan!";
            break;
        }
        
        total_error = error_vector.squaredNorm();
        if (iter > 0 && total_error > last_error)
        {
            LOG(WARNING) << "total error increased: " << total_error << ", " << last_error;
            break;
        }

        last_error = total_error;

        // 更新相机位姿
        Sophus::SE3d delta_pose = Sophus::SE3d::exp(delta_x.head<6>());
        camera_pose_optimized = delta_pose * camera_pose_optimized;

        // 更新3D点坐标
        for(int i = 0; i < points_3d.size(); i++)
        {
            points_3d_optimized[i] += delta_x.segment<3>(6 + 3 * i);
        }

        LOG(ERROR) << "iteration " << iter << " total error: " << total_error;

        if (delta_x.norm() < 1e-6)
        {
            LOG(INFO) << "参数收敛,经过 " << iter << " 次迭代";
            converged = true;
            break;
        }
    }

    if (!converged)
    {
        LOG(WARNING) << "达到最大迭代次数，但是参数未收敛";
    }
}