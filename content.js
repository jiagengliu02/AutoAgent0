/* Edit this file to update project metadata, videos, and reported results.
 * Paths are relative to index.html. Leave src empty until a video is available.
 * Leave paper, code, and arxiv empty until their release URLs are ready.
 * Keep the static table fallbacks in index.html in sync when changing results.
 */
window.AUTOAGENT = {
  project: {
    authors: [
      { name: 'Jiageng Liu', affiliations: [1], url: 'https://jiagengliu02.github.io/' },
      { name: 'Seth Z. Zhao', affiliations: [1], email: 'sethzhao506@g.ucla.edu', corresponding: true, url: 'https://sethzhao506.github.io/' },
      { name: 'Zewei Zhou', affiliations: [1], email: 'zeweizhou@ucla.edu', corresponding: true, url: 'https://zewei-zhou.github.io/' },
      { name: 'Alejandro Huerta', affiliations: [1], url: 'https://www.linkedin.com/in/alejandro-huerta42/' },
      { name: 'Aidan Bayer-Calvert', affiliations: [1], url: 'https://www.linkedin.com/in/aidan-bayer-calvert/' },
      { name: 'Yuxin Bao', affiliations: [1], url: 'https://www.linkedin.com/in/rebecca-bao-c13752hz/' },
      { name: 'Haotian Lin', affiliations: [2], url: 'https://darthutopian.github.io/' },
      { name: 'Robert Spataru', affiliations: [1], url: 'https://www.linkedin.com/in/robert-spataru/' },
      { name: 'Zijun Cui', affiliations: [3], url: 'https://zijuncui.com/' },
      { name: 'Yining Hong', affiliations: [4], url: 'https://evelinehong.github.io/' },
      { name: 'Bolei Zhou', affiliations: [1], url: 'https://boleizhou.github.io/' },
      { name: 'Jiaqi Ma', affiliations: [1], url: 'https://mobility-lab.seas.ucla.edu/about/' }
    ],
    affiliations: [
      { id: 1, name: 'University of California, Los Angeles', short: 'UCLA', url: 'https://www.ucla.edu/', logo: 'assets/logos/ucla.svg' },
      { id: 2, name: 'Carnegie Mellon University', short: 'CMU', url: 'https://www.cmu.edu/', logo: 'assets/logos/cmu.svg' },
      { id: 3, name: 'The University of Texas at Dallas', short: 'UT Dallas', url: 'https://www.utdallas.edu/', logo: 'assets/logos/ut-dallas.svg?v=20261014' },
      { id: 4, name: 'Stanford University', short: 'Stanford', url: 'https://www.stanford.edu/', logo: 'assets/logos/stanford.png' }
    ],
    paper: 'https://arxiv.org/abs/2610.18866',
    code: 'https://github.com/jiagengliu02/AutoAgent0',
    arxiv: '',
    citation: `@misc{autoagent0,
  title = {{AutoAgent0}: An Agentic Runtime for Safe Closed-Loop Driving},
  author = {Liu, Jiageng and Zhao, Seth Z. and Zhou, Zewei and Huerta, Alejandro and Bayer-Calvert, Aidan and Bao, Yuxin and Lin, Haotian and Spataru, Robert and Cui, Zijun and Hong, Yining and Zhou, Bolei and Ma, Jiaqi},
  year = {2026},
  eprint = {2610.18866},
  archivePrefix = {arXiv},
  primaryClass = {cs.RO},
  url = {https://arxiv.org/abs/2610.18866}
}`
  },
  overview: { src: 'assets/videos/AutoAgent0_demo.mp4', poster: 'assets/figures/video-overview.webp?v=film-v02-slide8-fixed', title: 'AutoAgent0: An Agentic Runtime for Safe Closed-Loop Driving', filename: 'AutoAgent0_demo.mp4', posterIsVideoFrame: true },
  scenarios: [
    {"id": "construction-permutations", "label": "Construction variants", "title": "Adapting to unfamiliar construction layouts", "videos": {"fail1": {"src": "assets/videos/demonstrations/construction_variants_fail1.mp4", "poster": "assets/figures/demonstrations/construction_variants_fail1.png", "posterIsVideoFrame": true, "preload": "none"}, "fail2": {"src": "assets/videos/demonstrations/construction_variants_fail2.mp4", "poster": "assets/figures/demonstrations/construction_variants_fail2.png", "posterIsVideoFrame": true, "preload": "none"}, "success": {"src": "assets/videos/demonstrations/construction_variants_success.mp4", "poster": "assets/figures/demonstrations/construction_variants_success.png", "posterIsVideoFrame": true, "preload": "none"}}},
    {"id": "bad-parking", "label": "Bad parking", "title": "Navigating around improperly parked vehicles", "videos": {"fail1": {"src": "assets/videos/demonstrations/bad_parking_fail1.mp4", "poster": "assets/figures/demonstrations/bad_parking_fail1.png", "posterIsVideoFrame": true, "preload": "none"}, "fail2": {"src": "assets/videos/demonstrations/bad_parking_fail2.mp4", "poster": "assets/figures/demonstrations/bad_parking_fail2.png", "posterIsVideoFrame": true, "preload": "none"}, "success": {"src": "assets/videos/demonstrations/bad_parking_success.mp4", "poster": "assets/figures/demonstrations/bad_parking_success.png", "posterIsVideoFrame": true, "preload": "none"}}},
    {"id": "animals", "label": "Animals", "title": "Responding to animals on the road", "videos": {"fail1": {"src": "assets/videos/demonstrations/animals_fail1.mp4", "poster": "assets/figures/demonstrations/animals_fail1.png", "posterIsVideoFrame": true, "preload": "none"}, "fail2": {"src": "assets/videos/demonstrations/animals_fail2.mp4", "poster": "assets/figures/demonstrations/animals_fail2.png", "posterIsVideoFrame": true, "preload": "none"}, "success": {"src": "assets/videos/demonstrations/animals_success.mp4", "poster": "assets/figures/demonstrations/animals_success.png", "posterIsVideoFrame": true, "preload": "none"}}},
    {"id": "pedestrian-crowds", "label": "Pedestrian crowds", "title": "Navigating dense pedestrian interactions", "videos": {"fail1": {"src": "assets/videos/demonstrations/pedestrian_crowds_fail1.mp4", "poster": "assets/figures/demonstrations/pedestrian_crowds_fail1.png", "posterIsVideoFrame": true, "preload": "none"}, "fail2": {"src": "assets/videos/demonstrations/pedestrian_crowds_fail2.mp4", "poster": "assets/figures/demonstrations/pedestrian_crowds_fail2.png", "posterIsVideoFrame": true, "preload": "none"}, "success": {"src": "assets/videos/demonstrations/pedestrian_crowds_success.mp4", "poster": "assets/figures/demonstrations/pedestrian_crowds_success.png", "posterIsVideoFrame": true, "preload": "none"}}}
  ],
  navsafeCases: [
    { id: 'a', label: 'Case A', title: 'Responding to a stopped vehicle', views: [
      { title: 'Scene View', src: 'assets/videos/nav-safe/cam_A_03b66343e1ac5d68.mp4', poster: 'assets/videos/nav-safe/cam_A_03b66343e1ac5d68.webp' },
      { title: 'BEV View', src: 'assets/videos/nav-safe/compare_A_03b66343e1ac5d68.mp4', poster: 'assets/videos/nav-safe/compare_A_03b66343e1ac5d68.webp' }
    ] },
    { id: 'b', label: 'Case B', title: 'Staying within the drivable area', views: [
      { title: 'Scene View', src: 'assets/videos/nav-safe/cam_B_a18d2b32f8415373.mp4', poster: 'assets/videos/nav-safe/cam_B_a18d2b32f8415373.webp' },
      { title: 'BEV View', src: 'assets/videos/nav-safe/compare_B_a18d2b32f8415373.mp4', poster: 'assets/videos/nav-safe/compare_B_a18d2b32f8415373.webp' }
    ] }
  ],
  methodVideos: [
    { id: 'perception', title: 'Scene context refinement', detail: 'Camera + LiDAR + semantic validation. Runtime demonstration.', src: 'assets/videos/scene-refinement.mp4', poster: 'assets/figures/video-perception.webp', filename: 'scene-refinement.mp4', posterIsVideoFrame: true },
    { id: 'recovery', title: 'Verifier-guided recovery', detail: 'Select → parameterize → verify. Runtime demonstration.', src: 'assets/videos/recovery-planning.mp4', poster: 'assets/figures/video-recovery.webp', filename: 'recovery-planning.mp4', posterIsVideoFrame: true },
    { id: 'adaptation', title: 'Closed-loop adaptation', detail: 'Observe, revise, and verify again. Runtime demonstration.', src: 'assets/videos/closed-loop-adaptation.mp4', poster: 'assets/figures/video-adaptation.webp', filename: 'closed-loop-adaptation.mp4', posterIsVideoFrame: true }
  ],
  // Metrics in each split: driving score, success rate (%), harmonic mean.
  results: [
    { name: 'TCP', input: 'RGB', id: [24.7,39.1,30.3], gen: [24.5,31.4,27.5], avg: [24.6,35.3,28.9] },
    { name: 'Orion', input: 'RGB', id: [53.0,52.0,52.5], gen: [51.2,46.0,48.5], avg: [52.1,49.0,50.5] },
    { name: 'HiP-AD', input: 'RGB', id: [74.1,70.7,72.4], gen: [67.1,56.7,61.5], avg: [70.6,63.7,67.0] },
    { name: 'SimLingo', input: 'RGB', id: [82.6,79.3,80.9], gen: [71.7,55.0,62.2], avg: [77.2,67.2,71.6] },
    { name: 'AlignDrive', input: 'RGB', id: [74.7,75.7,75.2], gen: [68.6,62.7,65.5], avg: [71.7,69.2,70.4] },
    { name: 'BevAD', input: 'RGB', id: [87.4,83.3,85.3], gen: [82.3,68.7,74.9], avg: [84.9,76.0,80.1] },
    { name: 'TF++', input: 'RGB + LiDAR', id: [83.3,78.5,80.8], gen: [75.4,61.1,67.5], avg: [79.4,69.8,74.2] },
    { name: 'TFv6', input: 'RGB + LiDAR', id: [90.2,93.3,91.7], gen: [79.5,70.7,74.8], avg: [84.9,82.0,83.3] },
    { name: 'UniAD', input: 'RGB', id: [47.5,36.3,41.2], gen: [44.0,27.6,33.9], avg: [45.8,32.0,37.6], expert: true },
    { name: 'UniAD + AutoAgent0', input: 'RGB + LiDAR', id: [68.4,69.0,68.7], gen: [69.2,69.0,69.1], avg: [68.8,69.0,68.9], ours: true },
    { name: 'SparseDriveV2', input: 'RGB', id: [70.0,74.0,71.9], gen: [62.9,52.7,57.4], avg: [66.5,63.4,64.7], expert: true },
    { name: 'SparseDriveV2 + AutoAgent0', input: 'RGB + LiDAR', id: [80.7,85.7,83.1], gen: [79.2,81.0,80.1], avg: [80.0,83.4,81.7], ours: true },
    { name: 'BridgeDrive', input: 'RGB + LiDAR', id: [91.6,95.0,93.3], gen: [81.9,75.0,78.3], avg: [86.8,85.0,85.8], expert: true },
    { name: 'BridgeDrive + AutoAgent0', input: 'RGB + LiDAR', id: [91.7,95.3,93.5], gen: [85.7,85.3,85.5], avg: [88.7,90.3,89.5], ours: true }
  ],
  navsafe: [
    ['DrivoR',64.7,50.7,188.0,87.9,58.7,31.7,52.4,43.3],
    ['GTRS-Dense–V2-99',64.3,49.5,163.9,95.9,52.0,40.0,54.2,36.7],
    ['AutoVLA',51.1,29.5,186.0,65.4,33.0,15.0,29.7,36.7],
    ['RAP',47.7,27.3,220.8,48.4,32.3,14.2,28.8,22.2],
    ['MTDrive–SFT',48.8,31.8,173.8,94.5,30.3,15.0,35.8,44.4],
    ['LTF',53.0,27.7,163.7,97.8,31.7,25.0,28.5,15.6],
    ['LTF + AutoAgent0',74.8,60.5,160.0,78.8,66.0,45.8,63.0,52.2],
    ['DiffusionDrive',44.8,21.0,165.1,97.7,21.3,11.7,23.6,22.2],
    ['DiffusionDrive + AutoAgent0',70.9,52.3,160.2,72.3,52.0,37.5,58.8,48.9],
    ['SparseDriveV2',65.3,48.8,168.0,93.4,54.0,43.3,48.2,41.1],
    ['SparseDriveV2 + AutoAgent0',77.7,63.6,165.3,76.0,68.0,50.0,64.5,63.3]
  ]
};
