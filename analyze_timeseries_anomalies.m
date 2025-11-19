%% Advanced Analysis of Time Series Anomalies
% This script performs various anomaly detection techniques on the synthetic dataset
% and compares their performance

clear; close all; clc;

%% 1. Load Dataset
fprintf('Loading dataset...\n');
load('synthetic_timeseries_data.mat');

% Extract data
X = [data_table.Sensor1, data_table.Temperature, data_table.Vibration, data_table.Pressure];
y_true = data_table.Anomaly;
X_train = [normal_table.Sensor1, normal_table.Temperature, normal_table.Vibration, normal_table.Pressure];

fprintf('Dataset loaded: %d samples, %d features\n', size(X, 1), size(X, 2));
fprintf('Training samples (normal): %d\n', size(X_train, 1));
fprintf('True anomalies: %d (%.2f%%)\n\n', sum(y_true), 100*mean(y_true));

%% 2. Feature Engineering
fprintf('Extracting features...\n');

% Statistical features (rolling windows)
window_size = 100;
features = zeros(size(X, 1), 0);

for i = 1:size(X, 2)
    % Rolling statistics
    rolling_mean = movmean(X(:, i), window_size);
    rolling_std = movstd(X(:, i), window_size);
    rolling_max = movmax(X(:, i), window_size);
    rolling_min = movmin(X(:, i), window_size);
    
    % Derivatives
    diff_signal = [0; diff(X(:, i))];
    
    % Combine features
    features = [features, rolling_mean, rolling_std, rolling_max, rolling_min, diff_signal];
end

% Normalize features
features_norm = normalize(features);
features_train = normalize([normal_table.Sensor1, normal_table.Temperature, ...
    normal_table.Vibration, normal_table.Pressure]);

fprintf('Features extracted: %d features per sample\n\n', size(features, 2));

%% 3. Method 1: Statistical Threshold (Z-Score)
fprintf('=== Method 1: Z-Score Analysis ===\n');

% Calculate z-scores based on training data
mu_train = mean(X_train);
sigma_train = std(X_train);
z_scores = abs((X - mu_train) ./ sigma_train);
z_score_max = max(z_scores, [], 2);

% Threshold
threshold_z = 3;
y_pred_zscore = z_score_max > threshold_z;

% Evaluate
[precision_z, recall_z, f1_z] = evaluate_predictions(y_true, y_pred_zscore);
fprintf('Precision: %.4f, Recall: %.4f, F1-Score: %.4f\n\n', precision_z, recall_z, f1_z);

%% 4. Method 2: Isolation Forest
fprintf('=== Method 2: Isolation Forest ===\n');

% Train isolation forest on normal data
rng(42);
num_trees = 100;
sample_size = min(256, size(X_train, 1));

% Simple isolation forest implementation using decision trees
contamination = 0.01;
iforest_scores = compute_isolation_scores(X, X_train, num_trees, sample_size);

% Threshold based on training data scores
train_scores = compute_isolation_scores(X_train, X_train, num_trees, sample_size);
threshold_if = prctile(train_scores, 99);
y_pred_iforest = iforest_scores > threshold_if;

[precision_if, recall_if, f1_if] = evaluate_predictions(y_true, y_pred_iforest);
fprintf('Precision: %.4f, Recall: %.4f, F1-Score: %.4f\n\n', precision_if, recall_if, f1_if);

%% 5. Method 3: One-Class SVM
fprintf('=== Method 3: One-Class SVM ===\n');

% Sample data for faster training
sample_idx = randperm(size(X_train, 1), min(5000, size(X_train, 1)));
X_train_sample = X_train(sample_idx, :);

% Train one-class SVM
svm_model = fitcsvm(X_train_sample, ones(size(X_train_sample, 1), 1), ...
    'KernelFunction', 'rbf', 'Standardize', true, ...
    'Nu', 0.05);

% Predict
[~, svm_scores] = predict(svm_model, X);
y_pred_svm = svm_scores < 0;

[precision_svm, recall_svm, f1_svm] = evaluate_predictions(y_true, y_pred_svm);
fprintf('Precision: %.4f, Recall: %.4f, F1-Score: %.4f\n\n', precision_svm, recall_svm, f1_svm);

%% 6. Method 4: Autoencoder (PCA-based)
fprintf('=== Method 4: PCA Reconstruction Error ===\n');

% PCA on training data
[coeff, ~, ~, ~, explained] = pca(X_train);
num_components = find(cumsum(explained) > 95, 1);
fprintf('Using %d principal components (95%% variance)\n', num_components);

% Project and reconstruct
X_projected = X * coeff(:, 1:num_components);
X_reconstructed = X_projected * coeff(:, 1:num_components)';

% Reconstruction error
reconstruction_error = sqrt(sum((X - X_reconstructed).^2, 2));

% Threshold based on training reconstruction error
X_train_proj = X_train * coeff(:, 1:num_components);
X_train_recon = X_train_proj * coeff(:, 1:num_components)';
train_error = sqrt(sum((X_train - X_train_recon).^2, 2));
threshold_pca = mean(train_error) + 3*std(train_error);

y_pred_pca = reconstruction_error > threshold_pca;

[precision_pca, recall_pca, f1_pca] = evaluate_predictions(y_true, y_pred_pca);
fprintf('Precision: %.4f, Recall: %.4f, F1-Score: %.4f\n\n', precision_pca, recall_pca, f1_pca);

%% 7. Method 5: LSTM Autoencoder Simulation (Using simple prediction)
fprintf('=== Method 5: Time Series Prediction Error ===\n');

% Simple moving average prediction
prediction_window = 50;
predictions = movmean(X, [prediction_window, 0], 1);
prediction_error = sqrt(sum((X - predictions).^2, 2));

% Threshold
threshold_pred = prctile(prediction_error, 99);
y_pred_lstm = prediction_error > threshold_pred;

[precision_lstm, recall_lstm, f1_lstm] = evaluate_predictions(y_true, y_pred_lstm);
fprintf('Precision: %.4f, Recall: %.4f, F1-Score: %.4f\n\n', precision_lstm, recall_lstm, f1_lstm);

%% 8. Ensemble Method - Majority Voting
fprintf('=== Ensemble Method: Majority Voting ===\n');

votes = y_pred_zscore + y_pred_iforest + y_pred_svm + y_pred_pca + y_pred_lstm;
y_pred_ensemble = votes >= 3;

[precision_ens, recall_ens, f1_ens] = evaluate_predictions(y_true, y_pred_ensemble);
fprintf('Precision: %.4f, Recall: %.4f, F1-Score: %.4f\n\n', precision_ens, recall_ens, f1_ens);

%% 9. Visualization: Comparison of Methods
fprintf('Generating comparison visualizations...\n');

figure('Position', [50, 50, 1600, 900]);

% Plot 1: Time series with different detection methods
subplot(3,2,1);
plot_window = 1:10000;
plot(t(plot_window)/60, X(plot_window, 1), 'k-', 'LineWidth', 0.5);
hold on;
detected_idx = find(y_pred_zscore(plot_window));
true_idx = find(y_true(plot_window));
if ~isempty(detected_idx)
    scatter(t(plot_window(detected_idx))/60, X(plot_window(detected_idx), 1), ...
        20, 'r', 'filled', 'MarkerFaceAlpha', 0.5);
end
if ~isempty(true_idx)
    scatter(t(plot_window(true_idx))/60, X(plot_window(true_idx), 1), ...
        40, 'g', 'LineWidth', 1.5);
end
title('Z-Score Detection');
xlabel('Time (min)');
ylabel('Sensor 1');
legend('Signal', 'Detected', 'True', 'Location', 'best');
grid on;

subplot(3,2,2);
plot(t(plot_window)/60, X(plot_window, 1), 'k-', 'LineWidth', 0.5);
hold on;
detected_idx = find(y_pred_iforest(plot_window));
true_idx = find(y_true(plot_window));
if ~isempty(detected_idx)
    scatter(t(plot_window(detected_idx))/60, X(plot_window(detected_idx), 1), ...
        20, 'r', 'filled', 'MarkerFaceAlpha', 0.5);
end
if ~isempty(true_idx)
    scatter(t(plot_window(true_idx))/60, X(plot_window(true_idx), 1), ...
        40, 'g', 'LineWidth', 1.5);
end
title('Isolation Forest');
xlabel('Time (min)');
ylabel('Sensor 1');
grid on;

subplot(3,2,3);
plot(t(plot_window)/60, X(plot_window, 1), 'k-', 'LineWidth', 0.5);
hold on;
detected_idx = find(y_pred_svm(plot_window));
true_idx = find(y_true(plot_window));
if ~isempty(detected_idx)
    scatter(t(plot_window(detected_idx))/60, X(plot_window(detected_idx), 1), ...
        20, 'r', 'filled', 'MarkerFaceAlpha', 0.5);
end
if ~isempty(true_idx)
    scatter(t(plot_window(true_idx))/60, X(plot_window(true_idx), 1), ...
        40, 'g', 'LineWidth', 1.5);
end
title('One-Class SVM');
xlabel('Time (min)');
ylabel('Sensor 1');
grid on;

subplot(3,2,4);
plot(t(plot_window)/60, X(plot_window, 1), 'k-', 'LineWidth', 0.5);
hold on;
detected_idx = find(y_pred_pca(plot_window));
true_idx = find(y_true(plot_window));
if ~isempty(detected_idx)
    scatter(t(plot_window(detected_idx))/60, X(plot_window(detected_idx), 1), ...
        20, 'r', 'filled', 'MarkerFaceAlpha', 0.5);
end
if ~isempty(true_idx)
    scatter(t(plot_window(true_idx))/60, X(plot_window(true_idx), 1), ...
        40, 'g', 'LineWidth', 1.5);
end
title('PCA Reconstruction');
xlabel('Time (min)');
ylabel('Sensor 1');
grid on;

subplot(3,2,5);
plot(t(plot_window)/60, X(plot_window, 1), 'k-', 'LineWidth', 0.5);
hold on;
detected_idx = find(y_pred_lstm(plot_window));
true_idx = find(y_true(plot_window));
if ~isempty(detected_idx)
    scatter(t(plot_window(detected_idx))/60, X(plot_window(detected_idx), 1), ...
        20, 'r', 'filled', 'MarkerFaceAlpha', 0.5);
end
if ~isempty(true_idx)
    scatter(t(plot_window(true_idx))/60, X(plot_window(true_idx), 1), ...
        40, 'g', 'LineWidth', 1.5);
end
title('Prediction Error');
xlabel('Time (min)');
ylabel('Sensor 1');
grid on;

subplot(3,2,6);
plot(t(plot_window)/60, X(plot_window, 1), 'k-', 'LineWidth', 0.5);
hold on;
detected_idx = find(y_pred_ensemble(plot_window));
true_idx = find(y_true(plot_window));
if ~isempty(detected_idx)
    scatter(t(plot_window(detected_idx))/60, X(plot_window(detected_idx), 1), ...
        20, 'r', 'filled', 'MarkerFaceAlpha', 0.5);
end
if ~isempty(true_idx)
    scatter(t(plot_window(true_idx))/60, X(plot_window(true_idx), 1), ...
        40, 'g', 'LineWidth', 1.5);
end
title('Ensemble Method');
xlabel('Time (min)');
ylabel('Sensor 1');
grid on;

saveas(gcf, 'anomaly_detection_comparison.png');

%% 10. Performance Comparison
figure('Position', [100, 100, 1400, 600]);

methods = {'Z-Score', 'Isolation Forest', 'One-Class SVM', 'PCA Recon.', 'Pred. Error', 'Ensemble'};
precision_vals = [precision_z, precision_if, precision_svm, precision_pca, precision_lstm, precision_ens];
recall_vals = [recall_z, recall_if, recall_svm, recall_pca, recall_lstm, recall_ens];
f1_vals = [f1_z, f1_if, f1_svm, f1_pca, f1_lstm, f1_ens];

subplot(1,3,1);
bar(precision_vals);
set(gca, 'XTickLabel', methods, 'XTickLabelRotation', 45);
ylabel('Precision');
title('Precision Comparison');
ylim([0 1]);
grid on;

subplot(1,3,2);
bar(recall_vals);
set(gca, 'XTickLabel', methods, 'XTickLabelRotation', 45);
ylabel('Recall');
title('Recall Comparison');
ylim([0 1]);
grid on;

subplot(1,3,3);
bar(f1_vals);
set(gca, 'XTickLabel', methods, 'XTickLabelRotation', 45);
ylabel('F1-Score');
title('F1-Score Comparison');
ylim([0 1]);
grid on;

saveas(gcf, 'performance_metrics_comparison.png');

%% 11. Confusion Matrices
figure('Position', [150, 150, 1400, 800]);

methods_pred = {y_pred_zscore, y_pred_iforest, y_pred_svm, y_pred_pca, y_pred_lstm, y_pred_ensemble};

for i = 1:6
    subplot(2,3,i);
    cm = confusionchart(double(y_true), double(methods_pred{i}));
    cm.Title = methods{i};
    cm.XLabel = 'Predicted';
    cm.YLabel = 'True';
end

saveas(gcf, 'confusion_matrices.png');

%% 12. ROC-like Analysis (using scores)
figure('Position', [200, 200, 1400, 600]);

subplot(1,3,1);
normal_idx = find(~y_true);
anomaly_idx = find(y_true);
histogram(z_score_max(normal_idx), 50, 'FaceColor', 'b', 'FaceAlpha', 0.5, 'Normalization', 'probability');
hold on;
histogram(z_score_max(anomaly_idx), 50, 'FaceColor', 'r', 'FaceAlpha', 0.5, 'Normalization', 'probability');
xline(threshold_z, 'k--', 'LineWidth', 2);
title('Z-Score Distribution');
xlabel('Z-Score');
ylabel('Probability');
legend('Normal', 'Anomaly', 'Threshold');
grid on;

subplot(1,3,2);
histogram(reconstruction_error(normal_idx), 50, 'FaceColor', 'b', 'FaceAlpha', 0.5, 'Normalization', 'probability');
hold on;
histogram(reconstruction_error(anomaly_idx), 50, 'FaceColor', 'r', 'FaceAlpha', 0.5, 'Normalization', 'probability');
xline(threshold_pca, 'k--', 'LineWidth', 2);
title('Reconstruction Error Distribution');
xlabel('Error');
ylabel('Probability');
legend('Normal', 'Anomaly', 'Threshold');
grid on;

subplot(1,3,3);
histogram(prediction_error(normal_idx), 50, 'FaceColor', 'b', 'FaceAlpha', 0.5, 'Normalization', 'probability');
hold on;
histogram(prediction_error(anomaly_idx), 50, 'FaceColor', 'r', 'FaceAlpha', 0.5, 'Normalization', 'probability');
xline(threshold_pred, 'k--', 'LineWidth', 2);
title('Prediction Error Distribution');
xlabel('Error');
ylabel('Probability');
legend('Normal', 'Anomaly', 'Threshold');
grid on;

saveas(gcf, 'score_distributions.png');

%% 13. Summary Report
fprintf('\n=== FINAL PERFORMANCE SUMMARY ===\n');
fprintf('%-20s %10s %10s %10s\n', 'Method', 'Precision', 'Recall', 'F1-Score');
fprintf('%s\n', repmat('-', 1, 55));
for i = 1:length(methods)
    fprintf('%-20s %10.4f %10.4f %10.4f\n', methods{i}, ...
        precision_vals(i), recall_vals(i), f1_vals(i));
end
fprintf('%s\n', repmat('-', 1, 55));

% Best method
[~, best_idx] = max(f1_vals);
fprintf('\nBest performing method: %s (F1=%.4f)\n', methods{best_idx}, f1_vals(best_idx));

fprintf('\nAll analysis completed!\n');
fprintf('Generated files:\n');
fprintf('  - anomaly_detection_comparison.png\n');
fprintf('  - performance_metrics_comparison.png\n');
fprintf('  - confusion_matrices.png\n');
fprintf('  - score_distributions.png\n');

%% Helper Functions

function [precision, recall, f1] = evaluate_predictions(y_true, y_pred)
    % Calculate precision, recall, and F1-score
    TP = sum(y_true & y_pred);
    FP = sum(~y_true & y_pred);
    FN = sum(y_true & ~y_pred);
    
    precision = TP / (TP + FP + eps);
    recall = TP / (TP + FN + eps);
    f1 = 2 * (precision * recall) / (precision + recall + eps);
end

function scores = compute_isolation_scores(X, X_train, num_trees, sample_size)
    % Simplified isolation score computation
    n = size(X, 1);
    scores = zeros(n, 1);
    
    for i = 1:num_trees
        % Random sample from training data
        idx = randperm(size(X_train, 1), sample_size);
        X_sample = X_train(idx, :);
        
        % Random feature and split
        feat_idx = randi(size(X, 2));
        split_val = X_sample(randi(sample_size), feat_idx);
        
        % Compute anomaly score based on distance to split
        scores = scores + abs(X(:, feat_idx) - split_val);
    end
    
    scores = scores / num_trees;
end
