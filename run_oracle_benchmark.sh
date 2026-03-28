#!/bin/bash
export GOOGLE_API_KEY="${GOOGLE_API_KEY:?Set GOOGLE_API_KEY before running}"
export CHROME_PATH='/home/jiachengl/.local/bin/google-chrome'

cd /data/spiderman/jiachengl/NextGen-CAPTCHAs

./test_benchmark.sh \
    --llm google \
    --model gemini-3-flash \
    --max-output-tokens 32768 \
    --max-steps 50 \
    --port 7860 \
    --seed 0 \
    --headless \
    --isolate-puzzles \
    --no-server \
    --oracle draft/oracle_strategies.json \
    --puzzles '3D_Viewpoint:20,Backmost_Layer:20,Box_Folding:20,Color_Counting:20,Dice_Roll_Path:20,Dynamic_Jigsaw:20,Hole_Counting:20,Illusory_Ribbons:20,Layered_Stack:20,Mirror:11,Multi_Script:20,Occluded_Pattern_Counting:20,Red_Dot:20,Rotation_Match:20,Shadow_Direction:20,Shadow_Plausible:8,Spooky_Circle:20,Spooky_Circle_Grid:20,Spooky_Jigsaw:20,Spooky_Shape_Grid:20,Spooky_Size:20,Spooky_Text:20,Static_Jigsaw:20,Structure_From_Motion:20,Subway_Paths:20,Temporal_Object_Continuity:20,Trajectory_Recovery:20'
