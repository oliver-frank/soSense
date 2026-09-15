function run_spm12_vbm_slope_RE_GMV_same_TIV_sample()
% RUN_SPM12_VBM_SLOPE_RE_GMV_SAME_TIV_SAMPLE
% Multiple-regression VBM analysis in SPM12 for SOSENSE CAT12 outputs.
%
% Effect of interest:
%   slope_clus_re
%
% Covariates of no interest:
%   clus, age, female, total gray-matter volume (GMV)
%
% Two models are estimated:
%   1) 6 mm smoothing, GMV-adjusted
%   2) 8 mm smoothing, GMV-adjusted
%
% The script:
%   - takes the exact 22-subject sample from the previous TIV model;
%   - normalizes codes such as SOSE_1JGLP, SOSE1JGLP, and sub-SOSE1JGLP;
%   - aborts if the same 22 subjects cannot be reconstructed exactly;
%   - obtains total GMV from the CAT12 volume table or CAT XML reports;
%   - uses a multiple-regression design with overall-mean centering;
%   - uses absolute masking at 0.1, implicit masking, and no global scaling;
%   - estimates positive and negative t-contrasts for slope_clus_re;
%   - saves matching tables, design diagnostics, batches, and logs.
%
% Run in MATLAB:
%   run_spm12_vbm_slope_RE_GMV_same_TIV_sample
%
% INTERPRETATION:
%   Adjusting for total GMV tests whether a regional gray-matter association
%   remains after accounting for each subject's total gray-matter volume.
%   This is a relative/regional sensitivity model and may remove widespread
%   gray-matter effects.

%% ========================================================================
%  USER SETTINGS
%  ========================================================================
spm_root = '/zi/software/Matlab/spm12_mit_cat12';

% The user-specified path uses lower-case current_analyses, while the earlier
% CAT12 preprocessing output used Current_Analyses. Linux is case-sensitive,
% so both variants are checked explicitly.
covariate_file_candidates = {
    '/zi/flstorage/group_csp/analyses/paul.grube/projects/project_sosense/out/df_slope.csv'
    '/zi/flstorage/group_csp/analyses/paul.grube/projects/project_sosense/out/df_slope.csv'
    };

vbm_root_candidates = {
    '/zi/flstorage/group_csp/analyses/paul.grube/projects/project_sosense/out/VBM'
    '/zi/flstorage/group_csp/analyses/paul.grube/projects/project_sosense/out/VBM'
    };

smoothing_kernels_mm = [6 8];

% The previous TIV-adjusted analysis reportedly contained 22 subjects.
% This script uses that exact TIV model sample as an inclusion reference and
% aborts before SPM estimation if all 22 cannot be reconstructed.
expected_reference_n = 23;

% CAT12 recommends an absolute threshold around 0.1 for basic VBM models.
absolute_mask_threshold = 0.1;

% false prevents accidental overwriting of an existing SPM model.
% Set true only when you intentionally want to delete and rebuild both
% GMV-adjusted model directories.
overwrite_existing_models = false;

% If a nuisance covariate has zero variance in the MRI subsample, it cannot
% be estimated. When true, such a nuisance covariate is omitted and logged.
% slope_clus_re is never omitted; zero variance there causes an error.
drop_zero_variance_nuisance = true;

% Warn when the absolute Pearson correlation between slope_clus_re and the
% brain-size adjustment variable is at least this value. This is a diagnostic
% warning only and does not change the model.
correlation_warning_threshold = 0.50;

%% ========================================================================
%  RESOLVE PATHS AND INITIALIZE SPM/CAT12
%  ========================================================================
covariate_file = resolve_existing_file( ...
    covariate_file_candidates, 'df_slope.csv');
vbm_root = resolve_vbm_root(vbm_root_candidates);

if ~isfolder(spm_root) || ~exist(fullfile(spm_root, 'spm.m'), 'file')
    error('SPM12 was not found at: %s', spm_root);
end

stats_root = fullfile(vbm_root, ...
    'SPM_stats_slope_clus_re_GMV_adjusted_same_sample_as_TIV');
make_dir(stats_root);
log_dir = fullfile(stats_root, 'logs');
make_dir(log_dir);

log_file = fullfile(log_dir, ...
    ['SPM12_VBM_slope_RE_GMV_same_TIV_sample_' datestr(now, 'yyyymmdd_HHMMSS') '.log']);
diary(log_file);
diary_cleanup = onCleanup(@() diary('off')); %#ok<NASGU>

fprintf('\n============================================================\n');
fprintf('SPM12 VBM analysis started: %s\n', datestr(now));
fprintf('SPM root      : %s\n', spm_root);
fprintf('VBM root      : %s\n', vbm_root);
fprintf('Covariate file: %s\n', covariate_file);
fprintf('Statistics dir: %s\n', stats_root);
fprintf('============================================================\n\n');

addpath(spm_root);
cat12_dir = fullfile(spm_root, 'toolbox', 'cat12');
if isfolder(cat12_dir)
    addpath(cat12_dir);
end

spm('defaults', 'FMRI');
spm_get_defaults('cmdline', true);
spm_jobman('initcfg');

fprintf('SPM version: %s\n', spm('Ver'));
if exist('cat12', 'file')
    try
        fprintf('CAT version: %s\n', cat_version());
    catch
        fprintf('CAT directory: %s\n', cat12_dir);
    end
end

%% ========================================================================
%  READ CLINICAL/COVARIATE DATA
%  ========================================================================
clinical = readtable(covariate_file, 'VariableNamingRule', 'preserve');

if width(clinical) < 9
    error(['df_slope.csv contains only %d columns. At least 9 are required ', ...
           'for columns 1 and 7-9.'], width(clinical));
end

idx_code   = resolve_table_column(clinical, 'code',           1);
idx_slope  = resolve_required_table_column(clinical, 'slope_clus_re');
idx_clus   = resolve_table_column(clinical, 'clustering-coefficient_mean_all',           7);
idx_age    = resolve_table_column(clinical, 'age',            8);
idx_female = resolve_table_column(clinical, 'female',         9);

code_raw = string(clinical{:, idx_code});
match_code = normalize_subject_code(code_raw);

clinical_table = table( ...
    match_code, code_raw, ...
    numeric_table_column(clinical, idx_slope,  'slope_clus_re'), ...
    numeric_table_column(clinical, idx_clus,   'clustering-coefficient_mean_all'), ...
    numeric_table_column(clinical, idx_age,    'age'), ...
    numeric_table_column(clinical, idx_female, 'female'), ...
    'VariableNames', {'match_code', 'code_raw', 'slope_clus_re', ...
                      'clustering-coefficient_mean_all', 'age', 'female'});

valid_code = strlength(clinical_table.match_code) > 0;
if any(~valid_code)
    warning('%d rows have missing/empty subject codes and will be ignored.', ...
        sum(~valid_code));
    clinical_table = clinical_table(valid_code, :);
end

assert_unique_codes(clinical_table.match_code, 'df_slope.csv');

female_values = unique(clinical_table.female(isfinite(clinical_table.female)));
if ~all(ismember(female_values, [0 1]))
    warning(['The female column contains values other than 0 and 1: %s. ', ...
             'The script will still model it numerically.'], ...
        strjoin(string(female_values(:)'), ', '));
end

fprintf('Clinical rows with valid codes: %d\n', height(clinical_table));
fprintf('Resolved columns: code=%d, slope=%d, clus=%d, age=%d, female=%d\n', ...
    idx_code, idx_slope, idx_clus, idx_age, idx_female);

%% ========================================================================
%  READ CAT12 TOTAL GRAY-MATTER VOLUME
%  ========================================================================
volume_table = load_cat_gmv_table(vbm_root);
assert_unique_codes(volume_table.match_code, 'CAT12 volume data');
fprintf('CAT12 GMV records: %d\n', height(volume_table));

%% ========================================================================
%  LOAD THE EXACT SUBJECT SAMPLE FROM THE PREVIOUS TIV MODEL
%  ========================================================================
[reference_table, reference_file] = load_tiv_reference_sample( ...
    vbm_root, expected_reference_n);
fprintf('TIV reference sample: %s\n', reference_file);
fprintf('TIV reference subjects: %d\n', height(reference_table));

%% ========================================================================
%  RUN BOTH GMV-ADJUSTED MODELS
%  ========================================================================
summary_rows = {};
summary_index = 0;

for kernel_mm = smoothing_kernels_mm
    image_table = discover_smoothed_gm_images(vbm_root, kernel_mm);
    assert_unique_codes(image_table.match_code, ...
        sprintf('s%d GM images', kernel_mm));

    fprintf('\n------------------------------------------------------------\n');
    fprintf('%d mm smoothing: found %d GM images.\n', ...
        kernel_mm, height(image_table));

    % Use one row per subject from the already estimated TIV model.
    % This avoids redefining the MRI sample from the larger clinical file.
    matching_report = build_reference_matching_report( ...
        reference_table, clinical_table, image_table, volume_table);
    matching_file = fullfile(stats_root, ...
        sprintf('reference_sample_matching_s%d.tsv', kernel_mm));
    write_tsv(matching_report, matching_file);

    fprintf('Reference matching report: %s\n', matching_file);
    fprintf('TIV-reference subjects without clinical row: %d\n', ...
        sum(~matching_report.has_clinical));
    fprintf('TIV-reference subjects without sMRI: %d\n', ...
        sum(~matching_report.has_sMRI));
    fprintf('TIV-reference subjects without GMV: %d\n', ...
        sum(~matching_report.has_CAT_volume));

    adjustment_name = 'GMV';
    model_dir = fullfile(stats_root, sprintf('s%d_GMV_adjusted', kernel_mm));

    fprintf('\nModel: s%d, adjustment=GMV\n', kernel_mm);

    summary_index = summary_index + 1;
    try
        [model_status, model_info] = run_one_vbm_model( ...
            matching_report, kernel_mm, model_dir, ...
            absolute_mask_threshold, overwrite_existing_models, ...
            drop_zero_variance_nuisance, ...
            correlation_warning_threshold, expected_reference_n);

        summary_rows(summary_index, :) = { ...
            kernel_mm, string(adjustment_name), string(model_status), ...
            model_info.n_subjects, model_info.n_regressors, ...
            model_info.design_rank, model_info.slope_adjustment_r, ...
            string(model_dir), string(model_info.message)};
    catch ME
        warning('Model s%d/GMV failed:\n%s', kernel_mm, getReport(ME, 'extended', 'hyperlinks', 'off'));
        summary_rows(summary_index, :) = { ...
            kernel_mm, string(adjustment_name), "FAILED", ...
            NaN, NaN, NaN, NaN, string(model_dir), string(ME.message)};
    end
end

model_summary = cell2table(summary_rows, 'VariableNames', { ...
    'smoothing_mm', 'adjustment', 'status', 'n_subjects', ...
    'n_regressors', 'design_rank', 'r_slope_adjustment', ...
    'model_directory', 'message'});
summary_file = fullfile(stats_root, 'SPM12_VBM_GMV_model_summary.tsv');
write_tsv(model_summary, summary_file);

fprintf('\n============================================================\n');
fprintf('SPM12 VBM analysis finished: %s\n', datestr(now));
fprintf('Model summary: %s\n', summary_file);
fprintf('Log file     : %s\n', log_file);
fprintf('============================================================\n');

disp(model_summary);
end

%% ========================================================================
%  MODEL FUNCTION
%  ========================================================================
function [status, info] = run_one_vbm_model( ...
    matching_report, kernel_mm, model_dir, ...
    mask_threshold, overwrite_existing, drop_zero_variance_nuisance, ...
    correlation_warning_threshold, expected_reference_n)

info = struct('n_subjects', NaN, 'n_regressors', NaN, ...
    'design_rank', NaN, 'slope_adjustment_r', NaN, 'message', "");

adjustment_values = matching_report.GMV;
adjustment_column_name = 'GMV';

complete_covariates = ...
    isfinite(matching_report.slope_clus_re) & ...
    isfinite(matching_report.age) & ...
    isfinite(matching_report.female) & ...
    isfinite(adjustment_values);

included = matching_report.has_clinical & ...
    matching_report.has_sMRI & complete_covariates;
analysis_table = matching_report(included, :);
analysis_table.brain_size_adjustment = adjustment_values(included);
analysis_table = sortrows(analysis_table, 'match_code');

if height(analysis_table) ~= expected_reference_n
    excluded = matching_report(~included, :);
    error(['The exact TIV reference sample could not be reproduced: ', ...
           '%d of %d subjects are complete. Inspect %s. Missing IDs: %s'], ...
        height(analysis_table), expected_reference_n, ...
        fullfile(fileparts(model_dir), ...
            sprintf('reference_sample_matching_s%d.tsv', kernel_mm)), ...
        strjoin(excluded.match_code, ', '));
end

% Store the exact model sample before SPM runs.
sample_file = fullfile(model_dir, 'included_subjects.tsv');
prepare_model_directory(model_dir, overwrite_existing);
write_tsv(analysis_table, sample_file);

image_files = cellstr(analysis_table.image_file);
for i = 1:numel(image_files)
    if ~exist(image_files{i}, 'file')
        error('Image listed for analysis does not exist: %s', image_files{i});
    end
end

% Confirm that image geometry is mutually compatible.
volume_headers = spm_vol(char(add_volume_index(image_files)));
spm_check_orientations(volume_headers);

covariate_names = {'slope_clus_re', 'age', 'female', ...
                   adjustment_column_name};
covariate_values = { ...
    analysis_table.slope_clus_re, ...
    analysis_table.age, ...
    analysis_table.female, ...
    analysis_table.brain_size_adjustment};

% Drop nuisance variables that cannot be estimated because they are constant.
keep_covariate = true(1, numel(covariate_names));
dropped_names = strings(0, 1);
for c = 1:numel(covariate_names)
    values = double(covariate_values{c}(:));
    if (max(values) - min(values)) <= max(eps(max(abs(values))), eps)
        if c == 1
            error('slope_clus_re has zero variance in the MRI subsample.');
        elseif drop_zero_variance_nuisance
            keep_covariate(c) = false;
            dropped_names(end + 1, 1) = string(covariate_names{c}); %#ok<AGROW>
            warning('Dropping zero-variance nuisance covariate: %s', ...
                covariate_names{c});
        else
            error('Nuisance covariate %s has zero variance.', ...
                covariate_names{c});
        end
    end
end
covariate_names = covariate_names(keep_covariate);
covariate_values = covariate_values(keep_covariate);

% Design diagnostics use explicitly mean-centered covariates, mirroring SPM's
% Overall mean centering (iCC = 1).
centered_covariates = zeros(height(analysis_table), numel(covariate_names));
for c = 1:numel(covariate_names)
    values = double(covariate_values{c}(:));
    centered_covariates(:, c) = values - mean(values);
end
X_diagnostic = [centered_covariates, ones(height(analysis_table), 1)];
design_rank = rank(X_diagnostic);
number_regressors = size(X_diagnostic, 2);

if design_rank < number_regressors
    error(['The design matrix is rank-deficient (rank %d of %d). ', ...
           'Inspect the covariates for perfect collinearity.'], ...
        design_rank, number_regressors);
end

% Correlation matrix and a direct diagnostic for slope vs GMV.
covariate_matrix = cell2mat(cellfun( ...
    @(x) double(x(:)), covariate_values, 'UniformOutput', false));
correlation_matrix = corrcoef(covariate_matrix, 'Rows', 'pairwise');
correlation_table = array2table(correlation_matrix, ...
    'VariableNames', matlab.lang.makeValidName(covariate_names), ...
    'RowNames', covariate_names);
writetable(correlation_table, fullfile(model_dir, ...
    'covariate_correlations.tsv'), 'FileType', 'text', ...
    'Delimiter', '\t', 'WriteRowNames', true);

adjustment_position = find(strcmp(covariate_names, adjustment_column_name), 1);
if isempty(adjustment_position)
    slope_adjustment_r = NaN;
else
    slope_adjustment_r = correlation_matrix(1, adjustment_position);
    fprintf('Correlation slope_clus_re vs %s: r = %.4f\n', ...
        adjustment_column_name, slope_adjustment_r);
    if abs(slope_adjustment_r) >= correlation_warning_threshold
        warning(['|r(slope_clus_re, %s)| = %.3f. Inspect design ', ...
                 'orthogonality carefully before inference.'], ...
            adjustment_column_name, slope_adjustment_r);
    end
end

% Save a human-readable model specification.
diagnostic_file = fullfile(model_dir, 'model_diagnostics.txt');
fid = fopen(diagnostic_file, 'w');
if fid < 0
    error('Could not write: %s', diagnostic_file);
end
fprintf(fid, 'Smoothing kernel: %d mm\n', kernel_mm);
fprintf(fid, 'Adjustment variable: %s\n', adjustment_column_name);
fprintf(fid, 'Number of subjects: %d\n', height(analysis_table));
fprintf(fid, 'Number of design regressors including intercept: %d\n', ...
    number_regressors);
fprintf(fid, 'Design rank: %d\n', design_rank);
fprintf(fid, 'Covariates in SPM order: %s\n', ...
    strjoin(covariate_names, ', '));
fprintf(fid, 'Dropped zero-variance nuisance covariates: %s\n', ...
    strjoin(cellstr(dropped_names), ', '));
fprintf(fid, 'r(slope_clus_re, %s): %.8f\n', ...
    adjustment_column_name, slope_adjustment_r);
fprintf(fid, 'Absolute mask threshold: %.4f\n', mask_threshold);
fprintf(fid, 'Global scaling: none\n');
fprintf(fid, 'Covariate centering in SPM: overall mean\n');
fclose(fid);

%% Specify the SPM multiple-regression design.
clear matlabbatch
matlabbatch{1}.spm.stats.factorial_design.dir = {model_dir};
matlabbatch{1}.spm.stats.factorial_design.des.mreg.scans = ...
    add_volume_index(image_files);

mcov = struct('c', {}, 'cname', {}, 'iCC', {});
for c = 1:numel(covariate_names)
    mcov(c).c = double(covariate_values{c}(:));
    mcov(c).cname = covariate_names{c};
    mcov(c).iCC = 1;  % Overall mean centering
end
matlabbatch{1}.spm.stats.factorial_design.des.mreg.mcov = mcov;
matlabbatch{1}.spm.stats.factorial_design.des.mreg.incint = 1;

matlabbatch{1}.spm.stats.factorial_design.cov = struct( ...
    'c', {}, 'cname', {}, 'iCFI', {}, 'iCC', {});
matlabbatch{1}.spm.stats.factorial_design.multi_cov = struct( ...
    'files', {}, 'iCFI', {}, 'iCC', {});

matlabbatch{1}.spm.stats.factorial_design.masking.tm.tma.athresh = ...
    mask_threshold;
matlabbatch{1}.spm.stats.factorial_design.masking.im = 1;
matlabbatch{1}.spm.stats.factorial_design.masking.em = {''};

% GMV is entered explicitly as a covariate. Therefore do not additionally
% calculate image globals, perform grand-mean scaling, or use proportional
% normalization.
matlabbatch{1}.spm.stats.factorial_design.globalc.g_omit = 1;
matlabbatch{1}.spm.stats.factorial_design.globalm.gmsca.gmsca_no = 1;
matlabbatch{1}.spm.stats.factorial_design.globalm.glonorm = 1;

specification_batch_file = fullfile(model_dir, ...
    'batch_01_design_specification.mat');
save(specification_batch_file, 'matlabbatch');
spm_jobman('run', matlabbatch);

spm_mat_file = fullfile(model_dir, 'SPM.mat');
if ~exist(spm_mat_file, 'file')
    error('SPM design specification did not create: %s', spm_mat_file);
end

%% Estimate the model.
clear matlabbatch
matlabbatch{1}.spm.stats.fmri_est.spmmat = {spm_mat_file};
matlabbatch{1}.spm.stats.fmri_est.write_residuals = 0;
matlabbatch{1}.spm.stats.fmri_est.method.Classical = 1;
estimation_batch_file = fullfile(model_dir, ...
    'batch_02_model_estimation.mat');
save(estimation_batch_file, 'matlabbatch');
spm_jobman('run', matlabbatch);

%% Define positive and negative contrasts using the SPM Contrast Manager.
% Using the batch interface avoids MATLAB structure-assignment errors that can
% occur when SPM.xCon is constructed manually across SPM12 revisions.
loaded = load(spm_mat_file, 'SPM');
SPM = loaded.SPM;
column_names = string(SPM.xX.name);
slope_columns = find(contains(lower(column_names), ...
    lower("slope_clus_re")));

if numel(slope_columns) ~= 1
    error(['Expected exactly one slope_clus_re design column, found %d: %s'], ...
        numel(slope_columns), strjoin(column_names(slope_columns), ', '));
end

positive_weights = zeros(1, size(SPM.xX.X, 2));
positive_weights(slope_columns) = 1;
negative_weights = -positive_weights;

clear matlabbatch
matlabbatch{1}.spm.stats.con.spmmat = {spm_mat_file};
matlabbatch{1}.spm.stats.con.consess{1}.tcon.name = ...
    'slope_clus_re positive';
matlabbatch{1}.spm.stats.con.consess{1}.tcon.weights = positive_weights;
matlabbatch{1}.spm.stats.con.consess{1}.tcon.sessrep = 'none';
matlabbatch{1}.spm.stats.con.consess{2}.tcon.name = ...
    'slope_clus_re negative';
matlabbatch{1}.spm.stats.con.consess{2}.tcon.weights = negative_weights;
matlabbatch{1}.spm.stats.con.consess{2}.tcon.sessrep = 'none';
matlabbatch{1}.spm.stats.con.delete = 1;

contrast_batch_file = fullfile(model_dir, ...
    'batch_03_slope_clus_re_contrasts.mat');
save(contrast_batch_file, 'matlabbatch');
spm_jobman('run', matlabbatch);

if ~exist(fullfile(model_dir, 'spmT_0001.nii'), 'file') || ...
        ~exist(fullfile(model_dir, 'spmT_0002.nii'), 'file')
    error('SPM contrast estimation did not create both expected spmT maps.');
end

info.n_subjects = height(analysis_table);
info.n_regressors = number_regressors;
info.design_rank = design_rank;
info.slope_adjustment_r = slope_adjustment_r;
if isempty(dropped_names)
    info.message = "Model estimated; no nuisance covariates dropped.";
else
    info.message = "Model estimated; dropped constant nuisance: " + ...
        strjoin(dropped_names, ", ");
end
status = 'ESTIMATED';

fprintf('Estimated model: %s\n', model_dir);
fprintf('Subjects: %d; regressors: %d; design rank: %d\n', ...
    info.n_subjects, info.n_regressors, info.design_rank);
end

%% ========================================================================
%  MATCHING AND DATA-LOADING FUNCTIONS
%  ========================================================================
function image_table = discover_smoothed_gm_images(vbm_root, kernel_mm)
mri_dir = fullfile(vbm_root, 'mri');
if ~isfolder(mri_dir)
    error('CAT12 mri directory does not exist: %s', mri_dir);
end

pattern = sprintf('s%dmwp1*.nii', kernel_mm);
hits = dir(fullfile(mri_dir, pattern));
if isempty(hits)
    error('No images matching %s were found in %s.', pattern, mri_dir);
end

image_file = string(fullfile({hits.folder}', {hits.name}'));
subject_id_from_image = strings(numel(hits), 1);
for i = 1:numel(hits)
    token = regexp(hits(i).name, 'sub-(.*?)_T1w', 'tokens', 'once');
    if isempty(token)
        token = regexp(hits(i).name, 'sub-([A-Za-z0-9_-]+)', ...
            'tokens', 'once');
    end
    if isempty(token)
        error('Could not derive subject code from image: %s', hits(i).name);
    end
    subject_id_from_image(i) = string(token{1});
end

match_code = normalize_subject_code(subject_id_from_image);
image_table = table(match_code, subject_id_from_image, image_file);
image_table = sortrows(image_table, 'match_code');
end

function volume_table = load_cat_gmv_table(vbm_root)
volume_file = fullfile(vbm_root, 'CAT12_TIV_GM_WM_CSF_WMH.tsv');

if exist(volume_file, 'file')
    raw = readtable(volume_file, 'FileType', 'text', ...
        'Delimiter', '\t', 'VariableNamingRule', 'preserve');
    idx_subject = resolve_table_column(raw, 'subject_id', 1);
    idx_gm = resolve_table_column(raw, 'GM', 3);

    subject_id_from_cat = string(raw{:, idx_subject});
    match_code = normalize_subject_code(subject_id_from_cat);
    GMV = numeric_table_column(raw, idx_gm, 'GM');

    volume_table = table(match_code, subject_id_from_cat, GMV);
    valid = strlength(volume_table.match_code) > 0;
    volume_table = volume_table(valid, :);
    volume_table = sortrows(volume_table, 'match_code');
    fprintf('Using CAT12 GMV table: %s\n', volume_file);
    return;
end

% Fallback: derive total GMV directly from CAT XML files.
report_dir = fullfile(vbm_root, 'report');
xml_hits = dir(fullfile(report_dir, 'cat_*.xml'));
if isempty(xml_hits)
    error(['Neither CAT12_TIV_GM_WM_CSF_WMH.tsv nor cat_*.xml reports ', ...
           'were found below %s.'], vbm_root);
end
if ~exist('cat_io_xml', 'file')
    error(['CAT GMV table is missing and cat_io_xml is unavailable. ', ...
           'Check the CAT12 path.']);
end

n = numel(xml_hits);
subject_id_from_cat = strings(n, 1);
GMV = nan(n, 1);

for i = 1:n
    token = regexp(xml_hits(i).name, '^cat_(sub-[^_]+)', ...
        'tokens', 'once');
    if isempty(token)
        subject_id_from_cat(i) = string(xml_hits(i).name);
    else
        subject_id_from_cat(i) = string(token{1});
    end

    xml_file = fullfile(xml_hits(i).folder, xml_hits(i).name);
    xml = cat_io_xml(xml_file);
    volumes = double(xml.subjectmeasures.vol_abs_CGW(:)');
    if numel(volumes) >= 2 && isfinite(volumes(2))
        GMV(i) = volumes(2);
    end
end

match_code = normalize_subject_code(subject_id_from_cat);
volume_table = table(match_code, subject_id_from_cat, GMV);
volume_table = volume_table(strlength(volume_table.match_code) > 0, :);
volume_table = sortrows(volume_table, 'match_code');

fallback_file = fullfile(vbm_root, 'CAT12_GMV_extracted_for_SPM.tsv');
write_tsv(volume_table, fallback_file);
fprintf('Extracted CAT12 GMV from XML: %s\n', fallback_file);
end

function [reference_table, reference_file] = load_tiv_reference_sample(vbm_root, expected_n)
% Prefer included_subjects.tsv from an already estimated TIV-adjusted model.
hits = dir(fullfile(vbm_root, '**', 'included_subjects.tsv'));
best_table = table();
best_file = "";
best_score = -Inf;

for i = 1:numel(hits)
    candidate_file = fullfile(hits(i).folder, hits(i).name);
    if ~contains(lower(candidate_file), 'tiv')
        continue;
    end
    try
        candidate = readtable(candidate_file, 'FileType', 'text', ...
            'Delimiter', '\t', 'VariableNamingRule', 'preserve');
        candidate_ids = extract_reference_ids(candidate);
        n_unique = numel(unique(candidate_ids(strlength(candidate_ids) > 0)));
        score = 1000 * (n_unique == expected_n) + n_unique;
        if contains(lower(candidate_file), 's6_tiv_adjusted')
            score = score + 10;
        end
        if score > best_score
            best_score = score;
            best_table = table(candidate_ids, ...
                'VariableNames', {'match_code'});
            best_file = string(candidate_file);
        end
    catch ME
        warning('Could not inspect TIV sample manifest %s: %s', ...
            candidate_file, ME.message);
    end
end

% Fallback to an analysis_subjects.tsv manifest.
if isempty(best_table)
    hits = dir(fullfile(vbm_root, '**', 'analysis_subjects.tsv'));
    for i = 1:numel(hits)
        candidate_file = fullfile(hits(i).folder, hits(i).name);
        try
            candidate = readtable(candidate_file, 'FileType', 'text', ...
                'Delimiter', '\t', 'VariableNamingRule', 'preserve');
            candidate_ids = extract_reference_ids(candidate);
            n_unique = numel(unique(candidate_ids(strlength(candidate_ids) > 0)));
            score = 1000 * (n_unique == expected_n) + n_unique;
            if score > best_score
                best_score = score;
                best_table = table(candidate_ids, ...
                    'VariableNames', {'match_code'});
                best_file = string(candidate_file);
            end
        catch ME
            warning('Could not inspect reference manifest %s: %s', ...
                candidate_file, ME.message);
        end
    end
end

if isempty(best_table)
    error(['No prior TIV included_subjects.tsv or analysis_subjects.tsv ', ...
           'could be found below %s.'], vbm_root);
end

best_table = best_table(strlength(best_table.match_code) > 0, :);
best_table = unique(best_table, 'rows', 'stable');
best_table = sortrows(best_table, 'match_code');
assert_unique_codes(best_table.match_code, char(best_file));

if height(best_table) ~= expected_n
    error(['The best prior TIV/reference manifest contains %d unique ', ...
           'subjects, but %d were expected. Selected file: %s'], ...
        height(best_table), expected_n, best_file);
end

reference_table = best_table;
reference_file = char(best_file);
end

function ids = extract_reference_ids(tbl)
names = string(tbl.Properties.VariableNames);
priority = ["match_code", "canonical_id", "code_original", ...
    "code_raw", "code", "subject_id_from_image", "subject_id"];
idx = [];
for p = 1:numel(priority)
    normalized_names = lower(regexprep(names, '[^A-Za-z0-9]', ''));
    desired = lower(regexprep(priority(p), '[^A-Za-z0-9]', ''));
    found = find(normalized_names == desired, 1);
    if ~isempty(found)
        idx = found;
        break;
    end
end
if isempty(idx)
    error('No recognizable subject-ID column was found.');
end
ids = normalize_subject_code(string(tbl{:, idx}));
end

function report = build_reference_matching_report( ...
    reference_table, clinical_table, image_table, volume_table)
% One output row per subject from the previous TIV model.
report = reference_table;

[has_clinical, clinical_location] = ismember( ...
    report.match_code, clinical_table.match_code);
report.has_clinical = has_clinical;
report.code_raw = strings(height(report), 1);
report.slope_clus_re = nan(height(report), 1);
report.age = nan(height(report), 1);
report.female = nan(height(report), 1);
report.code_raw(has_clinical) = ...
    clinical_table.code_raw(clinical_location(has_clinical));
report.slope_clus_re(has_clinical) = ...
    clinical_table.slope_clus_re(clinical_location(has_clinical));
report.age(has_clinical) = ...
    clinical_table.age(clinical_location(has_clinical));
report.female(has_clinical) = ...
    clinical_table.female(clinical_location(has_clinical));

[has_sMRI, image_location] = ismember( ...
    report.match_code, image_table.match_code);
report.has_sMRI = has_sMRI;
report.image_file = strings(height(report), 1);
report.subject_id_from_image = strings(height(report), 1);
report.image_file(has_sMRI) = ...
    image_table.image_file(image_location(has_sMRI));
report.subject_id_from_image(has_sMRI) = ...
    image_table.subject_id_from_image(image_location(has_sMRI));

[has_volume, volume_location] = ismember( ...
    report.match_code, volume_table.match_code);
report.has_CAT_volume = has_volume;
report.GMV = nan(height(report), 1);
report.subject_id_from_cat = strings(height(report), 1);
report.GMV(has_volume) = volume_table.GMV(volume_location(has_volume));
report.subject_id_from_cat(has_volume) = ...
    volume_table.subject_id_from_cat(volume_location(has_volume));

report.complete_clinical_covariates = ...
    isfinite(report.slope_clus_re) & ...
    isfinite(report.age) & ...
    isfinite(report.female);
report.included_GMV_model = report.has_clinical & report.has_sMRI & ...
    report.complete_clinical_covariates & isfinite(report.GMV);
end

function report = build_matching_report(clinical_table, image_table, volume_table)
report = clinical_table;

[has_sMRI, image_location] = ismember(report.match_code, image_table.match_code);
report.has_sMRI = has_sMRI;
report.image_file = strings(height(report), 1);
report.subject_id_from_image = strings(height(report), 1);
report.image_file(has_sMRI) = image_table.image_file(image_location(has_sMRI));
report.subject_id_from_image(has_sMRI) = ...
    image_table.subject_id_from_image(image_location(has_sMRI));

[has_volume, volume_location] = ismember(report.match_code, ...
    volume_table.match_code);
report.has_CAT_volume = has_volume;
report.GMV = nan(height(report), 1);
report.subject_id_from_cat = strings(height(report), 1);
report.GMV(has_volume) = volume_table.GMV(volume_location(has_volume));
report.subject_id_from_cat(has_volume) = ...
    volume_table.subject_id_from_cat(volume_location(has_volume));

report.complete_clinical_covariates = ...
    isfinite(report.slope_clus_re) & ...
    isfinite(report.age) & ...
    isfinite(report.female);
report.included_GMV_model = report.has_sMRI & ...
    report.complete_clinical_covariates & isfinite(report.GMV);
end

%% ========================================================================
%  GENERAL HELPERS
%  ========================================================================
function prepare_model_directory(model_dir, overwrite_existing)
if isfolder(model_dir)
    contents = dir(model_dir);
    contents = contents(~ismember({contents.name}, {'.', '..'}));
    if ~isempty(contents)
        if overwrite_existing
            [ok, message] = rmdir(model_dir, 's');
            if ~ok
                error('Could not remove existing model directory %s: %s', ...
                    model_dir, message);
            end
        else
            error(['Model directory already contains files: %s\n', ...
                   'Set overwrite_existing_models = true to rebuild it.'], ...
                model_dir);
        end
    end
end
make_dir(model_dir);
end

function make_dir(directory_name)
if ~isfolder(directory_name)
    [ok, message] = mkdir(directory_name);
    if ~ok
        error('Could not create directory %s: %s', directory_name, message);
    end
end
end

function indexed_files = add_volume_index(files)
files = cellstr(string(files));
indexed_files = cellfun(@(x) [x ',1'], files, 'UniformOutput', false);
indexed_files = indexed_files(:);
end

function normalized = normalize_subject_code(raw_codes)
% Canonical SOSENSE ID normalizer.
% Handles, for example:
%   SOSE_1JGLP, SOSE1JGLP, sub-SOSE1JGLP, 1JGLP,
%   SOSE_1JGLP_1 and file/path strings containing a SOSENSE ID.
raw = upper(strtrim(string(raw_codes)));
raw(ismissing(raw)) = "";
normalized = strings(size(raw));

for i = 1:numel(raw)
    value = raw(i);
    if strlength(value) == 0
        continue;
    end

    compact = regexprep(value, '[^A-Z0-9]', '');
    compact = regexprep(compact, '^SUB', '');

    % Prefer the canonical pattern SOSE plus exactly five ID characters.
    token = regexp(char(compact), 'SOSE[A-Z0-9]{5}', 'match', 'once');
    if ~isempty(token)
        normalized(i) = string(token);
        continue;
    end

    % Some source tables store only the five-character participant token.
    if ~isempty(regexp(char(compact), '^[A-Z0-9]{5}$', 'once'))
        normalized(i) = "SOSE" + compact;
        continue;
    end

    normalized(i) = compact;
end
end

function assert_unique_codes(codes, source_description)
codes = string(codes);
[unique_codes, ~, group_index] = unique(codes);
counts = accumarray(group_index, 1);
duplicates = unique_codes(counts > 1 & strlength(unique_codes) > 0);
if ~isempty(duplicates)
    error('Duplicate normalized subject codes in %s: %s', ...
        source_description, strjoin(duplicates, ', '));
end
end

function index = resolve_required_table_column(tbl, desired_name)
% Resolve a column strictly by name. No positional fallback is allowed.
% No positional fallback is permitted for the effect-of-interest column.
variable_names = string(tbl.Properties.VariableNames);
normalized_names = lower(regexprep(variable_names, '[^A-Za-z0-9]', ''));
normalized_desired = lower(regexprep(string(desired_name), ...
    '[^A-Za-z0-9]', ''));
match = find(normalized_names == normalized_desired);

if numel(match) == 1
    index = match;
elseif isempty(match)
    error(['Required column %s was not found in df_slope.csv. ', ...
           'The analysis will not fall back to another column. ', ...
           'Available columns: %s'], ...
        desired_name, strjoin(variable_names, ', '));
else
    error('Required column name %s is ambiguous.', desired_name);
end
end

function index = resolve_table_column(tbl, desired_name, fallback_index)
variable_names = string(tbl.Properties.VariableNames);
normalized_names = lower(regexprep(variable_names, '[^A-Za-z0-9]', ''));
normalized_desired = lower(regexprep(string(desired_name), ...
    '[^A-Za-z0-9]', ''));
match = find(normalized_names == normalized_desired);

if numel(match) == 1
    index = match;
elseif isempty(match)
    if width(tbl) < fallback_index
        error('Column %s was not found and fallback column %d is absent.', ...
            desired_name, fallback_index);
    end
    index = fallback_index;
    warning('Column %s not found by name; using column %d (%s).', ...
        desired_name, fallback_index, variable_names(fallback_index));
else
    error('Column name %s is ambiguous.', desired_name);
end
end

function values = numeric_table_column(tbl, column_index, column_description)
raw = tbl{:, column_index};
if isnumeric(raw) || islogical(raw)
    values = double(raw);
else
    values = str2double(string(raw));
end
values = values(:);

nonmissing_raw = ~ismissing(string(raw));
failed_conversion = nonmissing_raw & ~isfinite(values);
if any(failed_conversion)
    warning('%d values in %s could not be converted to finite numbers.', ...
        sum(failed_conversion), column_description);
end
end

function path_out = resolve_existing_file(candidates, description)
path_out = '';
for i = 1:numel(candidates)
    if exist(candidates{i}, 'file')
        path_out = candidates{i};
        break;
    end
end
if isempty(path_out)
    error('%s was not found. Checked:\n  %s', ...
        description, strjoin(candidates, '\n  '));
end
end


function path_out = resolve_vbm_root(candidates)
% Prefer a candidate that contains the expected CAT12 smoothed GM images.
path_out = '';
for i = 1:numel(candidates)
    candidate = candidates{i};
    if isfolder(fullfile(candidate, 'mri'))
        has_s6 = ~isempty(dir(fullfile(candidate, 'mri', 's6mwp1*.nii')));
        has_s8 = ~isempty(dir(fullfile(candidate, 'mri', 's8mwp1*.nii')));
        if has_s6 || has_s8
            path_out = candidate;
            return;
        end
    end
end

% Fall back to the first existing candidate to provide a clearer downstream
% error message about any missing CAT12 outputs.
path_out = resolve_existing_dir(candidates, 'VBM root directory');
end

function path_out = resolve_existing_dir(candidates, description)
path_out = '';
for i = 1:numel(candidates)
    if isfolder(candidates{i})
        path_out = candidates{i};
        break;
    end
end
if isempty(path_out)
    error('%s was not found. Checked:\n  %s', ...
        description, strjoin(candidates, '\n  '));
end
end

function write_tsv(tbl, output_file)
output_dir = fileparts(output_file);
if ~isempty(output_dir)
    make_dir(output_dir);
end
writetable(tbl, output_file, 'FileType', 'text', 'Delimiter', '\t');
end
