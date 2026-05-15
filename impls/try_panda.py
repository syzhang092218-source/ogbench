import ogbench
import cv2  # Import OpenCV

dataset_name = 'cube-double-play-singletask-task2-v0'
env = ogbench.make_env_and_datasets(dataset_name, env_only=True)

ob, info = env.reset()

done = False
while not done:
    action = env.action_space.sample()
    ob, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated

    frame = env.render()

    print('OBS:', ob.shape)
    print('Action:', action.shape)
    print('INFO:', info)

    # 1. Ensure the frame isn't None
    if frame is not None:
        # 2. Convert RGB (MuJoCo default) to BGR (OpenCV default)
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        # 3. Display the image in a window
        cv2.imshow('OGBench Environment', frame_bgr)

        # 4. Process GUI events and wait 1ms.
        # (If you want it to pause until you press a key, change 1 to 0)
        cv2.waitKey(1)

        # breakpoint() <-- Comment this out to let the video play, or leave it to step frame-by-frame

success = info['success']
print(f"Agent success: {success}")

# Clean up the window when the loop finishes
cv2.destroyAllWindows()