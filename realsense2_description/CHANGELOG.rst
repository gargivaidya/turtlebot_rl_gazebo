^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Changelog for package realsense2_description
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

3.1.7 (2020-06-16)
------------------

3.1.6 (2020-05-29)
------------------
* Merge branch 'revert-upstream-update' into 'ferrum-devel'
  Revert "Merge branch 'update-upstream' into 'erbium-devel'"
  See merge request ros-overlays/realsense!20
* Revert "Merge branch 'update-upstream' into 'erbium-devel'"
  This reverts merge request !17
* Contributors: procopiostein

3.1.5 (2020-05-18)
------------------
* Merge branch 'update-upstream' into 'erbium-devel'
  Update upstream
  See merge request ros-overlays/realsense!17
* fixed tests usage of extrinsic param
* fixed usage of use_nominal_extrinsics
* Merge branch 'ari-sim' into 'erbium-devel'
  updated realsense urdf for ari simulation
  See merge request ros-overlays/realsense!18
* updated realsense urdf for ari simulation
* Merge pull request #1016 from doronhi/fix_urdf_tests
  fix urdf tests
* Merge branch 'tykurtz-rename-frames-urdf' into development
* Merge branch 'parametrized-xacro' into development
* Contributors: Procópio Stein, doronhi, federiconardi, procopiostein

3.1.4 (2020-03-25)
------------------

3.1.3 (2020-03-25)
------------------

3.1.2 (2020-02-18)
------------------
* Merge branch 'topics-update' into 'erbium-devel'
  Changed /depth/points to /depth/color/points
  See merge request ros-overlays/realsense!12
* Changed /depth/points to /depth/color/points
* Contributors: saikishor, saracooper

3.1.1 (2020-02-05)
------------------

3.1.0 (2020-01-30)
------------------
* Merge branch 'pointcloud_gazebo' into 'erbium-devel'
  added publish_pointcloud argument to urdf and remove tab spaces
  See merge request ros-overlays/realsense!10
* added publish_pointcloud argument to urdf and remove tab spaces
* Contributors: Sai Kishor Kothakota, Victor Lopez

3.0.2 (2020-01-07)
------------------
* Merge branch 'fix-link-chain' into 'erbium-devel'
  Add pose_frame link
  See merge request ros-overlays/realsense!9
* Add transform between pose_frame and camera link for image orientation
* Add pose_frame link
* Contributors: Victor Lopez

3.0.1 (2019-12-16)
------------------

3.0.0 (2019-12-09)
------------------
* Merge branch 'gazebo_xacro' into 'upstream'
  Gazebo xacro
  See merge request ros-overlays/realsense!7
* added T265 URDF and gazebo xacro
* added D435 gazebo xacro to use realsense gazebo plugin
* upgrade version: 2.2.9
* Merge pull request #957 from saikishor/multiple-cameras
  added support for multiple cameras
* added support for multiple cameras
* Merge pull request #850 from peci1/patch-2
  Dependency on xacro missing.
* upgrade version: 2.2.8
* Dependency on xacro missing.
* update version to 2.2.7
* update version to 2.2.6
* Merge branch 'athackst-feature/realsense2_description' into development
* Merge branch 'feature/realsense2_description' of https://github.com/athackst/realsense into athackst-feature/realsense2_description
* changed install to single block
* updated references to realsense2_description
* moved description files into realsense2_description package
* Contributors: Allison Thackston, Martin Pecka, Sai Kishor Kothakota, Victor Lopez, doronhi
