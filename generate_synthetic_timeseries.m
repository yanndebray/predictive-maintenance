%% Generate Synthetic Time Series Dataset for Anomaly Detection
% This script creates a synthetic time series dataset with various types
% of anomalies for testing anomaly detection algorithms

clear; close all; clc;

%% Configuration
rng(42); % Set random seed for reproducibility

% Time series parameters
fs = 100;                    % Sampling frequency (Hz)
duration = 3600;             % Duration in seconds (1 hour)
t = (0:1/fs:duration-1/fs)'; % Time vector
N = length(t);

%% 1. Generate Normal Operation Signal
% Base signal: combination of trends, seasonality, and noise

% Trend component (slow drift)
trend = 0.001 * t;

% Seasonal components (multiple frequencies)
seasonal1 = 2 * sin(2*pi*0.01*t);      % 10-second period
seasonal2 = 1.5 * sin(2*pi*0.05*t);    % 2-second period
seasonal3 = 0.8 * cos(2*pi*0.002*t);   % 50-second period

% Random noise
noise = 0.3 * randn(N, 1);

% Combine components
normal_signal = 10 + trend + seasonal1 + seasonal2 + seasonal3 + noise;

%% 2. Inject Anomalies
anomaly_signal = normal_signal;
anomaly_labels = zeros(N, 1); % 0 = normal, 1 = anomaly

% Type 1: Point Anomalies (Spikes)
num_spikes = 15;
spike_indices = randperm(N, num_spikes);
for i = 1:num_spikes
    idx = spike_indices(i);
    if idx > N*0.1 && idx < N*0.9 % Avoid edges
        spike_magnitude = randsign() * (5 + 3*rand());
        anomaly_signal(idx) = anomaly_signal(idx) + spike_magnitude;
        anomaly_labels(idx) = 1;
    end
end

% Type 2: Contextual Anomalies (Level Shifts)
num_shifts = 5;
for i = 1:num_shifts
    shift_start = randi([round(N*0.1), round(N*0.8)]);
    shift_duration = randi([50, 200]);
    shift_end = min(shift_start + shift_duration, N);
    shift_magnitude = randsign() * (2 + 2*rand());
    
    anomaly_signal(shift_start:shift_end) = ...
        anomaly_signal(shift_start:shift_end) + shift_magnitude;
    anomaly_labels(shift_start:shift_end) = 1;
end

% Type 3: Collective Anomalies (Oscillations)
num_oscillations = 3;
for i = 1:num_oscillations
    osc_start = randi([round(N*0.1), round(N*0.7)]);
    osc_duration = randi([100, 300]);
    osc_end = min(osc_start + osc_duration, N);
    osc_freq = 0.5 + 0.5*rand(); % Random frequency
    osc_amplitude = 2 + 1.5*rand();
    
    t_osc = (0:osc_end-osc_start)' / fs;
    oscillation = osc_amplitude * sin(2*pi*osc_freq*t_osc);
    anomaly_signal(osc_start:osc_end) = ...
        anomaly_signal(osc_start:osc_end) + oscillation;
    anomaly_labels(osc_start:osc_end) = 1;
end

% Type 4: Gradual Drift Anomalies
num_drifts = 2;
for i = 1:num_drifts
    drift_start = randi([round(N*0.1), round(N*0.6)]);
    drift_duration = randi([200, 500]);
    drift_end = min(drift_start + drift_duration, N);
    drift_slope = randsign() * (0.01 + 0.005*rand());
    
    t_drift = (0:drift_end-drift_start)';
    drift = drift_slope * t_drift;
    anomaly_signal(drift_start:drift_end) = ...
        anomaly_signal(drift_start:drift_end) + drift;
    anomaly_labels(drift_start:drift_end) = 1;
end

%% 3. Create Multi-variate Dataset
% Add correlated sensor readings

% Sensor 2: Temperature (correlated with main signal)
temperature = 20 + 0.3*anomaly_signal + 0.5*randn(N, 1);

% Sensor 3: Vibration (partially correlated)
vibration = 5 + 0.15*anomaly_signal + sin(2*pi*0.08*t) + 0.4*randn(N, 1);

% Sensor 4: Pressure (independent with own anomalies)
pressure = 100 + 2*sin(2*pi*0.003*t) + 0.8*randn(N, 1);
% Add some pressure anomalies
pressure_anomaly_idx = randperm(N, 10);
pressure(pressure_anomaly_idx) = pressure(pressure_anomaly_idx) + randsign(10, 1) .* (3 + 2*rand(10, 1));

%% 4. Create Structured Dataset
% Create timetable for easy handling
time_stamps = datetime(2024, 1, 1, 0, 0, 0) + seconds(t);

data_table = timetable(time_stamps, anomaly_signal, temperature, vibration, pressure, anomaly_labels, ...
    'VariableNames', {'Sensor1', 'Temperature', 'Vibration', 'Pressure', 'Anomaly'});

% Also create normal signal table for training
normal_table = timetable(time_stamps, normal_signal, ...
    20 + 0.3*normal_signal + 0.5*randn(N, 1), ...
    5 + 0.15*normal_signal + sin(2*pi*0.08*t) + 0.4*randn(N, 1), ...
    100 + 2*sin(2*pi*0.003*t) + 0.8*randn(N, 1), ...
    'VariableNames', {'Sensor1', 'Temperature', 'Vibration', 'Pressure'});

%% 5. Save Dataset
save('synthetic_timeseries_data.mat', 'data_table', 'normal_table', 'fs', 't', ...
    'anomaly_signal', 'normal_signal', 'anomaly_labels', 'temperature', 'vibration', 'pressure');

fprintf('Dataset generated successfully!\n');
fprintf('Total samples: %d\n', N);
fprintf('Anomalies: %d (%.2f%%)\n', sum(anomaly_labels), 100*sum(anomaly_labels)/N);
fprintf('Duration: %.2f minutes\n', duration/60);
fprintf('Sampling frequency: %d Hz\n', fs);

%% 6. Visualize the Dataset
figure('Position', [100, 100, 1400, 800]);

% Plot 1: Main signal with anomalies
subplot(4,1,1);
plot(t/60, normal_signal, 'b-', 'LineWidth', 0.5, 'DisplayName', 'Normal');
hold on;
plot(t/60, anomaly_signal, 'r-', 'LineWidth', 0.5, 'DisplayName', 'With Anomalies');
anomaly_times = t(anomaly_labels == 1)/60;
anomaly_values = anomaly_signal(anomaly_labels == 1);
scatter(anomaly_times, anomaly_values, 20, 'filled', 'MarkerFaceColor', [0.8 0 0], ...
    'DisplayName', 'Anomaly Points', 'MarkerFaceAlpha', 0.3);
ylabel('Sensor 1');
title('Synthetic Time Series with Anomalies');
legend('Location', 'best');
grid on;

% Plot 2: Temperature
subplot(4,1,2);
plot(t/60, temperature, 'g-', 'LineWidth', 0.5);
hold on;
scatter(anomaly_times, temperature(anomaly_labels == 1), 20, 'r', 'filled', 'MarkerFaceAlpha', 0.3);
ylabel('Temperature (°C)');
grid on;

% Plot 3: Vibration
subplot(4,1,3);
plot(t/60, vibration, 'm-', 'LineWidth', 0.5);
hold on;
scatter(anomaly_times, vibration(anomaly_labels == 1), 20, 'r', 'filled', 'MarkerFaceAlpha', 0.3);
ylabel('Vibration');
grid on;

% Plot 4: Pressure
subplot(4,1,4);
plot(t/60, pressure, 'c-', 'LineWidth', 0.5);
hold on;
scatter(anomaly_times, pressure(anomaly_labels == 1), 20, 'r', 'filled', 'MarkerFaceAlpha', 0.3);
ylabel('Pressure (kPa)');
xlabel('Time (minutes)');
grid on;

% Save figure
saveas(gcf, 'synthetic_timeseries_visualization.png');

%% 7. Generate Statistics Report
figure('Position', [150, 150, 1200, 600]);

% Anomaly distribution
subplot(2,3,1);
histogram(anomaly_labels, 'FaceColor', [0.2 0.4 0.8]);
title('Anomaly Distribution');
xlabel('Label (0=Normal, 1=Anomaly)');
ylabel('Count');
grid on;

% Signal histogram
subplot(2,3,2);
histogram(normal_signal, 30, 'FaceColor', 'b', 'FaceAlpha', 0.5, 'DisplayName', 'Normal');
hold on;
histogram(anomaly_signal, 30, 'FaceColor', 'r', 'FaceAlpha', 0.5, 'DisplayName', 'With Anomalies');
title('Signal Value Distribution');
xlabel('Value');
ylabel('Frequency');
legend;
grid on;

% Autocorrelation
subplot(2,3,3);
autocorr(normal_signal, 'NumLags', 200);
title('Autocorrelation (Normal Signal)');

% Power spectral density
subplot(2,3,4);
[pxx, f] = pwelch(normal_signal, [], [], [], fs);
plot(f, 10*log10(pxx), 'LineWidth', 1.5);
title('Power Spectral Density');
xlabel('Frequency (Hz)');
ylabel('Power/Frequency (dB/Hz)');
grid on;
xlim([0 1]);

% Correlation matrix
subplot(2,3,5);
corr_matrix = corrcoef([anomaly_signal, temperature, vibration, pressure]);
imagesc(corr_matrix);
colorbar;
title('Sensor Correlation Matrix');
set(gca, 'XTick', 1:4, 'XTickLabel', {'S1', 'Temp', 'Vib', 'Press'});
set(gca, 'YTick', 1:4, 'YTickLabel', {'S1', 'Temp', 'Vib', 'Press'});
axis square;

% Anomaly size distribution
subplot(2,3,6);
anomaly_magnitudes = abs(anomaly_signal(anomaly_labels==1) - normal_signal(anomaly_labels==1));
histogram(anomaly_magnitudes, 30, 'FaceColor', [0.8 0.2 0.2]);
title('Anomaly Magnitude Distribution');
xlabel('Magnitude');
ylabel('Count');
grid on;

saveas(gcf, 'synthetic_timeseries_statistics.png');

fprintf('\nVisualization saved successfully!\n');
fprintf('Files created:\n');
fprintf('  - synthetic_timeseries_data.mat\n');
fprintf('  - synthetic_timeseries_visualization.png\n');
fprintf('  - synthetic_timeseries_statistics.png\n');

%% Helper function
function s = randsign(varargin)
    % Generate random signs (+1 or -1)
    if nargin == 0
        s = 2 * (rand() > 0.5) - 1;
    else
        s = 2 * (rand(varargin{:}) > 0.5) - 1;
    end
end

fprintf('Statistics report generated successfully!\n');

