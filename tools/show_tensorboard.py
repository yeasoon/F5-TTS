from tensorboard.backend.event_processing import event_accumulator

# Path to your event file or directory
# ea = event_accumulator.EventAccumulator('/data/tts/F5-TTS/runs/F5TTS_v1_Base/events.out.tfevents.1762428574.yxx-opt-fng9l-51639-worker-0.1546876.0')
ea = event_accumulator.EventAccumulator('/data/tts/F5-TTS/runs/F5TTS_v1_Base/events.out.tfevents.1762742207.yxx-opt-b6l26-1533759-worker-0.6718.0')
ea.Reload()

# List available tags (scalars, histograms, images, etc.)
print(ea.Tags())

# Extract a scalar, e.g. training loss
loss_events = ea.Scalars('loss')

# for e in loss_events:
#     print(f"Step {e.step}: value={e.value}")
import matplotlib.pyplot as plt

steps = [e.step for e in loss_events]
values = [e.value for e in loss_events]
plt.figure(figsize=(60,40))
plt.plot(steps, values, label='train/loss')
plt.xlabel('Step')
plt.ylabel('Loss')
plt.title('Training Loss')
plt.grid(True)
plt.savefig("train_loss.png", dpi=300, bbox_inches='tight')