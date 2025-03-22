import pyrealsense2 as rs

# try:
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 6)
pipeline.start(config)

while True:
    frames = pipeline.wait_for_frames()
    depth = frames.get_depth_frame()
    if not depth: continue

    coverage = [0] * 128
    for y in range(720):
        for x in range(1280):
            dist = depth.get_distance(x, y)
            if 0 < dist and dist < 1:
                coverage[x//10] += 1
        
        if y%20 == 19:
            line = ""
            for c in coverage:
                line += " .:nhBXWW"[c//25]
            coverage = [0] * 128
            print(line)
exit(0)

# except Exception as e:
#     print(f"error: {e}")
#     pass