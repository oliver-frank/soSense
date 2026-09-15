function run_spm12_vbm_slope_analysis()
% RUN_SPM12_VBM_SLOPE_ANALYSIS
% -------------------------------------------------------------------------
% Cross-sectional CAT12/SPM12 voxel-based morphometry analysis.
%
% Primary association:
%   local grey-matter volume ~ slope_clus_ols + clus + age + female + TIV
%
% The script:
%   1) reads the behavioural/covariate CSV;
%   2) discovers CAT12 s6mwp1 and s8mwp1 grey-matter images;
%   3) reads TIV from the CAT12 TIV table (or CAT XML reports as fallback);
%   4) matches subjects despite formats such as SOSE_1JGLP, SOSE1JGLP,
%      and sub-SOSE1JGLP;
%   5) excludes and documents subjects without all required data;
%   6) specifies and estimates separate 6 mm and 8 mm SPM12 multiple-
%      regression models;
%   7) creates positive, negative, and two-sided/F contrasts for
%      slope_clus_ols;
%   8) writes matched-subject and design-diagnostic tables.
%
% Run in MATLAB:
%   run_spm12_vbm_slope_analysis
%
% Final model per smoothing kernel:
%   GMV = intercept + slope_clus_ols + clus + age + female + TIV + error
%
% Notes:
%   - All regressors are overall-mean centred by SPM.
%   - TIV is entered as a nuisance covariate; no proportional scaling is
%     performed.
%   - Absolute masking threshold is 0.1.
%   - Only complete cases with an image and TIV are analysed.

%% ========================================================================
% USER SETTINGS
% =========================================================================
spm_root = ...
    '/zi/software/Matlab/spm12_mit_cat12';

vbm_root = ...
    '/zi/flstorage/group_csp/analyses/paul.grube/projects/project_sosense/out/VBM';

covariate_file = ...
    '/zi/flstorage/group_csp/analyses/paul.grube/projects/project_sosense/out/df_slope.csv';
 
% The preprocessing script writes this table.
tiv_file = fullfile(vbm_root, 'CAT12_TIV_GM_WM_CSF_WMH.tsv');

% CAT12 smoothed GM maps are expected below this directory.
mri_dir = fullfile(vbm_root, 'mri');

% Separate model folders will be generated for 6 mm and 8 mm.
stats_root = fullfile(vbm_root, 'SPM_stats_slope_clus_ols');
smoothing_kernels = [6 8];

% Use the same subjects in both models. Recommended for direct comparison.
require_all_kernels = true;

% CAT12's recommended starting point for absolute masking of VBM data.
absolute_mask_threshold = 0.1;

% Safety switch. If false, an existing SPM.mat is never overwritten.
overwrite_existing_models = true;

% Create and estimate the SPM models and contrasts.
run_model_estimation = true;

%% ========================================================================
% INITIAL SETUP
% =========================================================================
validate_required_path(spm_root, 'directory', 'SPM12 directory');
validate_required_path(vbm_root, 'directory', 'VBM root directory');
validate_required_path(mri_dir, 'directory', 'CAT12 MRI output directory');

% Linux paths are case-sensitive. Try the corresponding Current_Analyses
% spelling if the user-supplied lowercase path is not found.
if ~isfile(covariate_file)
    alternative_covariate_file = strrep(covariate_file, ...
        '/current_analyses/', '/Current_Analyses/');
    if isfile(alternative_covariate_file)
        warning('Using case-adjusted covariate path: %s', ...
            alternative_covariate_file);
        covariate_file = alternative_covariate_file;
    else
        error('Covariate file not found: %s', covariate_file);
    end
end

if ~isfile(fullfile(spm_root, 'spm.m'))
    error('spm.m was not found directly below: %s', spm_root);
end

addpath(spm_root);
cat12_dir = fullfile(spm_root, 'toolbox', 'cat12');
if isfolder(cat12_dir)
    addpath(cat12_dir);
end

spm('defaults', 'FMRI');
spm_get_defaults('cmdline', true);
spm_jobman('initcfg');

make_dir(stats_root);
log_dir = fullfile(stats_root, 'logs');
make_dir(log_dir);
log_file = fullfile(log_dir, ...
    ['SPM12_VBM_slope_' datestr(now, 'yyyymmdd_HHMMSS') '.log']);
diary(log_file);
diary_cleanup = onCleanup(@() diary('off')); %#ok<NASGU>

fprintf('\n============================================================\n');
fprintf('SPM12 VBM analysis started: %s\n', datestr(now));
fprintf('SPM root      : %s\n', spm_root);
fprintf('VBM root      : %s\n', vbm_root);
fprintf('Covariate file: %s\n', covariate_file);
fprintf('MRI directory : %s\n', mri_dir);
fprintf('Statistics    : %s\n', stats_root);
fprintf('SPM version   : %s\n', spm('Ver'));
fprintf('============================================================\n\n');

%% ========================================================================
% LOAD COVARIATES, TIV, AND IMAGES
% =========================================================================
covariates = read_covariate_table(covariate_file);
tiv = read_tiv_table_or_xml(tiv_file, fullfile(vbm_root, 'report'));

image_tables = cell(numel(smoothing_kernels), 1);
for k = 1:numel(smoothing_kernels)
    image_tables{k} = discover_smoothed_images( ...
        mri_dir, smoothing_kernels(k));
end

fprintf('Covariate rows after duplicate check: %d\n', height(covariates));
fprintf('TIV rows after duplicate check       : %d\n', height(tiv));
for k = 1:numel(smoothing_kernels)
    fprintf('s%d image rows                       : %d\n', ...
        smoothing_kernels(k), height(image_tables{k}));
end

%% ========================================================================
% BUILD MATCHED DATASET AND EXCLUSION REPORT
% =========================================================================
[analysis_table, exclusion_table] = build_analysis_table( ...
    covariates, tiv, image_tables, smoothing_kernels, ...
    require_all_kernels);

exclusion_file = fullfile(stats_root, 'subject_matching_and_exclusions.tsv');
writetable(exclusion_table, exclusion_file, ...
    'FileType', 'text', 'Delimiter', '\t');

master_file = fullfile(stats_root, 'analysis_subjects_all_kernels.tsv');
writetable(analysis_table, master_file, ...
    'FileType', 'text', 'Delimiter', '\t');

fprintf('\nSubjects retained for analysis: %d\n', height(analysis_table));
fprintf('Matching/exclusion report     : %s\n', exclusion_file);
fprintf('Master analysis table         : %s\n', master_file);

if height(analysis_table) < 8
    error(['Only %d complete subjects remain. The model has five ', ...
        'regressors plus an intercept and should not be estimated.'], ...
        height(analysis_table));
end

% Validate the shared covariate design before running any voxelwise model.
covariate_names = {'slope_clus_ols', 'age', 'female', 'TIV'};
X = [analysis_table.slope_clus_ols, ...
     analysis_table.age, analysis_table.female, analysis_table.TIV];
validate_design_matrix(X, covariate_names, stats_root);

female_values = unique(analysis_table.female);
if ~all(ismember(female_values, [0 1]))
    warning(['female is not coded exclusively as 0/1. Values found: %s. ', ...
        'The variable will be treated as a numeric covariate.'], ...
        mat2str(female_values'));
end

%% ========================================================================
% SPECIFY, ESTIMATE, AND CONTRAST EACH SMOOTHING KERNEL
% =========================================================================
for k = 1:numel(smoothing_kernels)
    kernel = smoothing_kernels(k);
    image_var = sprintf('image_s%d', kernel);

    % If require_all_kernels=false, remove rows lacking this kernel here.
    current_table = analysis_table;
    has_current_image = strlength(current_table.(image_var)) > 0;
    current_table = current_table(has_current_image, :);

    if height(current_table) < 8
        warning('Skipping s%d model: only %d complete subjects.', ...
            kernel, height(current_table));
        continue;
    end

    model_dir = fullfile(stats_root, sprintf('s%dmm', kernel));
    prepare_model_directory(model_dir, overwrite_existing_models);

    model_subject_file = fullfile(model_dir, 'analysis_subjects.tsv');
    writetable(current_table, model_subject_file, ...
        'FileType', 'text', 'Delimiter', '\t');

    fprintf('\n------------------------------------------------------------\n');
    fprintf('Preparing %d mm model with N = %d\n', ...
        kernel, height(current_table));
    fprintf('Model directory: %s\n', model_dir);

    scans = cellstr(current_table.(image_var));
    scans = add_volume_index(scans);

    X_current = [current_table.slope_clus_ols,  ...
                 current_table.age, current_table.female, ...
                 current_table.TIV];

    save(fullfile(model_dir, 'model_inputs.mat'), ...
        'current_table', 'X_current', 'covariate_names', 'scans', ...
        'absolute_mask_threshold', 'kernel');

    clear matlabbatch

    % --------------------------------------------------------------------
    % 1. Multiple-regression design specification
    % --------------------------------------------------------------------
    matlabbatch{1}.spm.stats.factorial_design.dir = {model_dir};
    matlabbatch{1}.spm.stats.factorial_design.des.mreg.scans = scans;

    for j = 1:numel(covariate_names)
        matlabbatch{1}.spm.stats.factorial_design.des.mreg.mcov(j).c = ...
            X_current(:, j);
        matlabbatch{1}.spm.stats.factorial_design.des.mreg.mcov(j).cname = ...
            covariate_names{j};
        % 1 = overall-mean centering in SPM12.
        matlabbatch{1}.spm.stats.factorial_design.des.mreg.mcov(j).iCC = 1;
    end

    matlabbatch{1}.spm.stats.factorial_design.des.mreg.incint = 1;

    % No additional factorial-design covariates outside the mreg block.
    matlabbatch{1}.spm.stats.factorial_design.cov = struct( ...
        'c', {}, 'cname', {}, 'iCFI', {}, 'iCC', {});
    matlabbatch{1}.spm.stats.factorial_design.multi_cov = struct( ...
        'files', {}, 'iCFI', {}, 'iCC', {});

    % Absolute tissue masking for modulated-normalized GM maps.
    matlabbatch{1}.spm.stats.factorial_design.masking.tm.tma.athresh = ...
        absolute_mask_threshold;
    matlabbatch{1}.spm.stats.factorial_design.masking.im = 1;
    matlabbatch{1}.spm.stats.factorial_design.masking.em = {''};

    % TIV is explicitly modelled above. Do not calculate or scale by an
    % image global value.
    matlabbatch{1}.spm.stats.factorial_design.globalc.g_omit = 1;
    matlabbatch{1}.spm.stats.factorial_design.globalm.gmsca.gmsca_no = 1;
    matlabbatch{1}.spm.stats.factorial_design.globalm.glonorm = 1;

    % --------------------------------------------------------------------
    % 2. Classical SPM model estimation
    % --------------------------------------------------------------------
    matlabbatch{2}.spm.stats.fmri_est.spmmat = ...
        {fullfile(model_dir, 'SPM.mat')};
    matlabbatch{2}.spm.stats.fmri_est.write_residuals = 0;
    matlabbatch{2}.spm.stats.fmri_est.method.Classical = 1;

    save(fullfile(model_dir, 'SPM_design_and_estimation_batch.mat'), ...
        'matlabbatch');

    if run_model_estimation
        spm_jobman('run', matlabbatch);
        create_interest_contrasts(model_dir, 'slope_clus_ols');
        export_spm_design_information(model_dir);
    else
        fprintf(['Model estimation disabled. Batch saved but not run: ', ...
            '%s\n'], model_dir);
    end
end

fprintf('\n============================================================\n');
fprintf('SPM12 VBM analysis finished: %s\n', datestr(now));
fprintf('Statistics root: %s\n', stats_root);
fprintf('Log file       : %s\n', log_file);
fprintf('============================================================\n');
end

%% ========================================================================
% LOCAL FUNCTIONS
% =========================================================================
function validate_required_path(path_name, path_type, label)
switch lower(path_type)
    case 'directory'
        exists_flag = isfolder(path_name);
    case 'file'
        exists_flag = isfile(path_name);
    otherwise
        error('Unknown path type: %s', path_type);
end
if ~exists_flag
    error('%s not found: %s', label, path_name);
end
end

function make_dir(directory_name)
if ~isfolder(directory_name)
    [ok, message] = mkdir(directory_name);
    if ~ok
        error('Could not create directory %s: %s', ...
            directory_name, message);
    end
end
end

function indexed_files = add_volume_index(files)
indexed_files = cellfun(@(x) [char(x) ',1'], files, ...
    'UniformOutput', false);
indexed_files = indexed_files(:);
end

function ids = canonicalize_ids(raw_ids)
ids = upper(strtrim(string(raw_ids)));
% Remove the BIDS entity prefix so that sub-SOSE1JGLP matches SOSE_1JGLP.
ids = regexprep(ids, '^SUB[-_]?', '');
ids = regexprep(ids, '[^A-Z0-9]', '');
end

function tbl = read_covariate_table(filename)
raw = read_table_preserve_names(filename, ',');
if width(raw) < 9
    error(['Covariate file has only %d columns; columns 1, 2, and 7-9 ', ...
        'are required: %s'], width(raw), filename);
end

code_col   = find_column(raw, 'code', 1);
slope_col  = find_column(raw, 'slope_clus_ols', 2);
age_col    = find_column(raw, 'age', 8);
female_col = find_column(raw, 'female', 9);

code = string(raw{:, code_col});
if size(code, 2) > 1
    code = code(:, 1);
end

tbl = table;
tbl.code_original = code;
tbl.canonical_id = canonicalize_ids(code);
tbl.slope_clus_ols = to_numeric_vector(raw{:, slope_col}, ...
    'slope_clus_ols');
tbl.age = to_numeric_vector(raw{:, age_col}, 'age');
tbl.female = to_numeric_vector(raw{:, female_col}, 'female');

invalid_code = ismissing(tbl.code_original) | strlength(tbl.canonical_id) == 0;
if any(invalid_code)
    warning('Dropping %d row(s) with missing/invalid code.', sum(invalid_code));
    tbl(invalid_code, :) = [];
end

tbl = collapse_duplicate_rows(tbl, ...
    {'slope_clus_ols', 'age', 'female'}, 'covariate');
tbl = sortrows(tbl, 'canonical_id');
end

function raw = read_table_preserve_names(filename, delimiter)
% Explicitly declare tabular inputs as text files. Older MATLAB releases do
% not recognize the .tsv extension automatically and otherwise stop before
% applying the requested delimiter.
try
    raw = readtable(filename, ...
        'FileType', 'text', ...
        'Delimiter', delimiter, ...
        'VariableNamingRule', 'preserve');
catch ME_new
    try
        % Compatibility with MATLAB versions predating VariableNamingRule.
        raw = readtable(filename, ...
            'FileType', 'text', ...
            'Delimiter', delimiter, ...
            'PreserveVariableNames', true);
    catch ME_old
        error(['Could not read table: %s\nModern readtable error: %s\n', ...
            'Legacy readtable error: %s'], ...
            filename, ME_new.message, ME_old.message);
    end
end
end

function idx = find_column(tbl, desired_name, fallback_position)
names = string(tbl.Properties.VariableNames);
idx = find(strcmpi(strtrim(names), desired_name), 1);
if isempty(idx)
    idx = fallback_position;
    warning(['Column "%s" was not found by name. Using column %d ', ...
        '(%s) as specified.'], desired_name, idx, names(idx));
end
end

function x = to_numeric_vector(values, variable_name)
if isnumeric(values) || islogical(values)
    x = double(values);
else
    x = str2double(strrep(strtrim(string(values)), ',', '.'));
end
x = x(:);

% Distinguish genuine missing values from non-numeric text where possible.
if all(isnan(x)) && ~all(ismissing(string(values)))
    warning('All values of %s became NaN after numeric conversion.', ...
        variable_name);
end
end

function tbl = read_tiv_table_or_xml(tiv_filename, report_dir)
if isfile(tiv_filename)
    raw = read_table_preserve_names(tiv_filename, '\t');
    subject_col = find_column(raw, 'subject_id', 1);
    tiv_col = find_column(raw, 'TIV', 2);

    subject_id = string(raw{:, subject_col});
    if size(subject_id, 2) > 1
        subject_id = subject_id(:, 1);
    end

    tbl = table;
    tbl.tiv_subject_id = subject_id;
    tbl.canonical_id = canonicalize_ids(subject_id);
    tbl.TIV = to_numeric_vector(raw{:, tiv_col}, 'TIV');
else
    warning(['TIV table not found; attempting to read CAT12 XML reports: ', ...
        '%s'], tiv_filename);
    if ~isfolder(report_dir)
        error('Neither TIV table nor CAT12 report directory was found.');
    end
    xml_hits = dir(fullfile(report_dir, 'cat_*.xml'));
    if isempty(xml_hits)
        error('No CAT XML reports found in: %s', report_dir);
    end
    if ~exist('cat_io_xml', 'file')
        error('cat_io_xml is unavailable; cannot extract TIV from XML.');
    end

    n = numel(xml_hits);
    subject_id = strings(n, 1);
    tiv_values = nan(n, 1);
    for i = 1:n
        [~, xml_name] = fileparts(xml_hits(i).name);
        token = regexp(xml_name, '^cat_(sub-[^_]+)', ...
            'tokens', 'once');
        if isempty(token)
            subject_id(i) = xml_name;
        else
            subject_id(i) = token{1};
        end
        xml_file = fullfile(xml_hits(i).folder, xml_hits(i).name);
        try
            xml = cat_io_xml(xml_file);
            tiv_values(i) = sum(double( ...
                xml.subjectmeasures.vol_abs_CGW(:)), 'omitnan');
        catch ME
            warning('Could not extract TIV from %s: %s', ...
                xml_file, ME.message);
        end
    end

    tbl = table(subject_id, canonicalize_ids(subject_id), tiv_values, ...
        'VariableNames', {'tiv_subject_id', 'canonical_id', 'TIV'});
end

invalid = strlength(tbl.canonical_id) == 0;
tbl(invalid, :) = [];
tbl = collapse_duplicate_rows(tbl, {'TIV'}, 'TIV');
tbl = sortrows(tbl, 'canonical_id');
end

function tbl = discover_smoothed_images(mri_dir, kernel)
pattern = sprintf('s%dmwp1*.nii', kernel);
hits = dir(fullfile(mri_dir, pattern));
if isempty(hits)
    error('No images matching %s were found in %s.', pattern, mri_dir);
end

n = numel(hits);
subject_id = strings(n, 1);
image_file = strings(n, 1);
for i = 1:n
    image_file(i) = fullfile(hits(i).folder, hits(i).name);
    token = regexp(hits(i).name, ...
        'mwp1(sub-[^_]+)(?:_T1w)?\.nii$', 'tokens', 'once');
    if isempty(token)
        token = regexp(hits(i).name, ...
            'mwp1(sub-[^_]+).*\.nii$', 'tokens', 'once');
    end
    if isempty(token)
        error('Could not derive subject ID from image: %s', hits(i).name);
    end
    subject_id(i) = token{1};
end

image_var = sprintf('image_s%d', kernel);
tbl = table(subject_id, canonicalize_ids(subject_id), image_file, ...
    'VariableNames', {'image_subject_id', 'canonical_id', image_var});
tbl = collapse_duplicate_rows(tbl, {image_var}, ...
    sprintf('s%d image', kernel));
tbl = sortrows(tbl, 'canonical_id');
end

function tbl = collapse_duplicate_rows(tbl, compare_variables, label)
[unique_ids, ~, group_idx] = unique(tbl.canonical_id, 'stable');
keep = false(height(tbl), 1);

for g = 1:numel(unique_ids)
    rows = find(group_idx == g);
    keep(rows(1)) = true;
    if numel(rows) == 1
        continue;
    end

    for v = 1:numel(compare_variables)
        variable_name = compare_variables{v};
        values = tbl.(variable_name)(rows, :);
        first_value = values(1, :);
        if isnumeric(values) || islogical(values)
            equal_values = all(all(isequaln_matrix(values, first_value)));
        else
            equal_values = all(string(values) == string(first_value));
        end
        if ~equal_values
            error(['Conflicting duplicate %s rows for canonical ID %s ', ...
                'in variable %s.'], label, unique_ids(g), variable_name);
        end
    end
    warning('Duplicate identical %s rows for %s; retaining the first.', ...
        label, unique_ids(g));
end

tbl = tbl(keep, :);
end

function tf = isequaln_matrix(values, first_value)
% Element-wise equality that treats corresponding NaNs as equal.
tf = (values == first_value) | (isnan(values) & isnan(first_value));
end

function [analysis, exclusions] = build_analysis_table( ...
    covariates, tiv, image_tables, kernels, require_all_kernels)

all_ids = [covariates.canonical_id; tiv.canonical_id];
for k = 1:numel(image_tables)
    all_ids = [all_ids; image_tables{k}.canonical_id]; %#ok<AGROW>
end
all_ids = unique(all_ids);
all_ids = sort(all_ids);

n = numel(all_ids);
code_original = strings(n, 1);
slope_clus_ols = nan(n, 1);
age = nan(n, 1);
female = nan(n, 1);
TIV = nan(n, 1);
has_covariate_row = false(n, 1);
has_TIV = false(n, 1);

image_values = cell(numel(kernels), 1);
has_image = cell(numel(kernels), 1);
for k = 1:numel(kernels)
    image_values{k} = strings(n, 1);
    has_image{k} = false(n, 1);
end

reason = strings(n, 1);

for i = 1:n
    id = all_ids(i);

    cidx = find(covariates.canonical_id == id, 1);
    if ~isempty(cidx)
        has_covariate_row(i) = true;
        code_original(i) = covariates.code_original(cidx);
        slope_clus_ols(i) = covariates.slope_clus_ols(cidx);
        age(i) = covariates.age(cidx);
        female(i) = covariates.female(cidx);
    end

    tidx = find(tiv.canonical_id == id, 1);
    if ~isempty(tidx) && isfinite(tiv.TIV(tidx))
        has_TIV(i) = true;
        TIV(i) = tiv.TIV(tidx);
    end

    for k = 1:numel(kernels)
        iidx = find(image_tables{k}.canonical_id == id, 1);
        image_var = sprintf('image_s%d', kernels(k));
        if ~isempty(iidx)
            image_values{k}(i) = image_tables{k}.(image_var)(iidx);
            has_image{k}(i) = isfile(image_values{k}(i));
        end
    end

    reasons_i = strings(0, 1);
    if ~has_covariate_row(i)
        reasons_i(end + 1) = 'missing_covariate_row'; %#ok<AGROW>
    else
        if ~isfinite(slope_clus_ols(i))
            reasons_i(end + 1) = 'missing_slope_clus_ols'; %#ok<AGROW>
        end

        if ~isfinite(age(i))
            reasons_i(end + 1) = 'missing_age'; %#ok<AGROW>
        end
        if ~isfinite(female(i))
            reasons_i(end + 1) = 'missing_female'; %#ok<AGROW>
        end
    end
    if ~has_TIV(i)
        reasons_i(end + 1) = 'missing_TIV'; %#ok<AGROW>
    end

    for k = 1:numel(kernels)
        if ~has_image{k}(i) && require_all_kernels
            reasons_i(end + 1) = sprintf('missing_s%d_image', kernels(k)); %#ok<AGROW>
        end
    end

    reason(i) = strjoin(reasons_i, ';');
end

exclusions = table(all_ids, code_original, has_covariate_row, ...
    slope_clus_ols, age, female, has_TIV, TIV, ...
    'VariableNames', {'canonical_id', 'code_original', ...
    'has_covariate_row', 'slope_clus_ols', 'age', 'female', ...
    'has_TIV', 'TIV'});

for k = 1:numel(kernels)
    exclusions.(sprintf('has_s%d_image', kernels(k))) = has_image{k};
    exclusions.(sprintf('image_s%d', kernels(k))) = image_values{k};
end
exclusions.exclusion_reason = reason;
exclusions.included = reason == "";

analysis = exclusions(exclusions.included, :);
analysis.exclusion_reason = [];
analysis.included = [];
analysis.has_covariate_row = [];
analysis.has_TIV = [];
for k = 1:numel(kernels)
    analysis.(sprintf('has_s%d_image', kernels(k))) = [];
end

% If models may use different samples, retain all otherwise-complete rows
% with at least one available image. Kernel-specific filtering occurs later.
if ~require_all_kernels
    complete_nonimage = isfinite(exclusions.slope_clus_ols) & ...
        isfinite(exclusions.age) & ...
        isfinite(exclusions.female) & exclusions.has_TIV;
    any_image = false(height(exclusions), 1);
    for k = 1:numel(kernels)
        any_image = any_image | exclusions.(sprintf('has_s%d_image', kernels(k)));
    end
    analysis = exclusions(complete_nonimage & any_image, :);
    analysis.exclusion_reason = [];
    analysis.included = [];
    analysis.has_covariate_row = [];
    analysis.has_TIV = [];
    for k = 1:numel(kernels)
        analysis.(sprintf('has_s%d_image', kernels(k))) = [];
    end
end

analysis = sortrows(analysis, 'canonical_id');
end

function validate_design_matrix(X, names, output_dir)
if any(~isfinite(X(:)))
    error('The final design matrix still contains non-finite values.');
end

n = size(X, 1);
p = size(X, 2);
means = mean(X, 1);
sds = std(X, 0, 1);
mins = min(X, [], 1);
maxs = max(X, [], 1);

constant_predictors = sds == 0;
if any(constant_predictors)
    error('Constant predictor(s) after matching: %s', ...
        strjoin(names(constant_predictors), ', '));
end

X_centered = X - means;
X_design = [ones(n, 1), X_centered];
design_rank = rank(X_design);
expected_rank = p + 1;
if design_rank < expected_rank
    error(['Design matrix is rank deficient: rank %d, expected %d. ', ...
        'Inspect covariate correlations and coding.'], ...
        design_rank, expected_rank);
end

Z = X_centered ./ sds;
condition_number = cond([ones(n, 1), Z]);
correlation_matrix = corrcoef(X);
vif = calculate_vif(X);

summary_table = table(string(names(:)), means(:), sds(:), mins(:), ...
    maxs(:), vif(:), 'VariableNames', ...
    {'variable', 'mean', 'sd', 'minimum', 'maximum', 'VIF'});
writetable(summary_table, fullfile(output_dir, ...
    'design_covariate_summary.tsv'), ...
    'FileType', 'text', 'Delimiter', '\t');

corr_table = array2table(correlation_matrix, ...
    'VariableNames', matlab.lang.makeValidName(names), ...
    'RowNames', names);
writetable(corr_table, fullfile(output_dir, ...
    'design_covariate_correlations.tsv'), ...
    'FileType', 'text', 'Delimiter', '\t', 'WriteRowNames', true);

fid = fopen(fullfile(output_dir, 'design_diagnostics.txt'), 'w');
if fid < 0
    error('Could not write design diagnostics.');
end
cleanup = onCleanup(@() fclose(fid)); %#ok<NASGU>
fprintf(fid, 'N = %d\n', n);
fprintf(fid, 'Predictors = %d plus intercept\n', p);
fprintf(fid, 'Design rank = %d of %d\n', design_rank, expected_rank);
fprintf(fid, 'Condition number (standardized design) = %.6f\n', ...
    condition_number);
fprintf(fid, '\nVariable\tMean\tSD\tMin\tMax\tVIF\n');
for j = 1:p
    fprintf(fid, '%s\t%.8g\t%.8g\t%.8g\t%.8g\t%.6f\n', ...
        names{j}, means(j), sds(j), mins(j), maxs(j), vif(j));
end

fprintf('\nDesign rank: %d/%d\n', design_rank, expected_rank);
fprintf('Standardized design condition number: %.3f\n', condition_number);
for j = 1:p
    fprintf('VIF %-16s: %.3f\n', names{j}, vif(j));
    if vif(j) >= 10
        warning('High multicollinearity: VIF for %s is %.2f.', ...
            names{j}, vif(j));
    elseif vif(j) >= 5
        warning('Moderately high VIF for %s: %.2f.', names{j}, vif(j));
    end
end
end

function vif = calculate_vif(X)
p = size(X, 2);
vif = nan(1, p);
for j = 1:p
    y = X(:, j);
    others = X(:, setdiff(1:p, j));
    design = [ones(size(X, 1), 1), others];
    beta = design \ y;
    residuals = y - design * beta;
    sse = sum(residuals .^ 2);
    sst = sum((y - mean(y)) .^ 2);
    r2 = 1 - sse / sst;
    vif(j) = 1 / max(1 - r2, eps);
end
end

function prepare_model_directory(model_dir, overwrite_existing)
if isfolder(model_dir)
    contents = dir(model_dir);
    names = {contents.name};
    nontrivial = ~ismember(names, {'.', '..'});
    if any(nontrivial)
        if overwrite_existing
            fprintf('Removing existing model directory: %s\n', model_dir);
            rmdir(model_dir, 's');
        else
            error(['The model directory is not empty: %s. Set ', ...
                'overwrite_existing_models=true to replace it.'], model_dir);
        end
    end
end
make_dir(model_dir);
end

function create_interest_contrasts(model_dir, interest_name)
spm_file = fullfile(model_dir, 'SPM.mat');
if ~isfile(spm_file)
    error('SPM.mat not found after model estimation: %s', spm_file);
end

loaded = load(spm_file, 'SPM');
SPM = loaded.SPM;
design_names = spm_design_names_to_string(SPM.xX.name);
interest_column = find_design_column(design_names, interest_name);

n_columns = size(SPM.xX.X, 2);
positive = zeros(1, n_columns);
positive(interest_column) = 1;
negative = -positive;

fprintf('SPM design columns for %s:\n', model_dir);
for i = 1:numel(design_names)
    fprintf('  %2d: %s\n', i, design_names(i));
end
fprintf('Interest column: %d (%s)\n', ...
    interest_column, design_names(interest_column));

clear matlabbatch
matlabbatch{1}.spm.stats.con.spmmat = {spm_file};
matlabbatch{1}.spm.stats.con.consess{1}.tcon.name = ...
    [interest_name ' positive'];
matlabbatch{1}.spm.stats.con.consess{1}.tcon.weights = positive;
matlabbatch{1}.spm.stats.con.consess{1}.tcon.sessrep = 'none';

matlabbatch{1}.spm.stats.con.consess{2}.tcon.name = ...
    [interest_name ' negative'];
matlabbatch{1}.spm.stats.con.consess{2}.tcon.weights = negative;
matlabbatch{1}.spm.stats.con.consess{2}.tcon.sessrep = 'none';

matlabbatch{1}.spm.stats.con.consess{3}.fcon.name = ...
    [interest_name ' any direction'];
matlabbatch{1}.spm.stats.con.consess{3}.fcon.weights = positive;
matlabbatch{1}.spm.stats.con.consess{3}.fcon.sessrep = 'none';
matlabbatch{1}.spm.stats.con.delete = 0;

save(fullfile(model_dir, 'SPM_contrast_batch.mat'), 'matlabbatch');
spm_jobman('run', matlabbatch);
end

function idx = find_design_column(design_names, target)
clean_names = strtrim(design_names);
idx = find(strcmpi(clean_names, target));
if isempty(idx)
    idx = find(endsWith(lower(clean_names), lower(target)));
end
if isempty(idx)
    idx = find(contains(lower(clean_names), lower(target)));
end
if numel(idx) ~= 1
    error(['Could not uniquely identify design column "%s". ', ...
        'Matches: %s'], target, mat2str(idx'));
end
end

function names = spm_design_names_to_string(raw_names)
if ischar(raw_names)
    names = string(cellstr(raw_names));
elseif iscell(raw_names)
    names = string(raw_names(:));
else
    names = string(raw_names(:));
end
names = names(:);
end

function export_spm_design_information(model_dir)
spm_file = fullfile(model_dir, 'SPM.mat');
loaded = load(spm_file, 'SPM');
SPM = loaded.SPM;

names = spm_design_names_to_string(SPM.xX.name);
X = SPM.xX.X;
column_names = matlab.lang.makeUniqueStrings( ...
    matlab.lang.makeValidName(cellstr(names)));

design_table = array2table(X, 'VariableNames', column_names);
writetable(design_table, fullfile(model_dir, ...
    'SPM_design_matrix.tsv'), ...
    'FileType', 'text', 'Delimiter', '\t');

name_table = table((1:numel(names))', names, ...
    'VariableNames', {'column', 'SPM_name'});
writetable(name_table, fullfile(model_dir, ...
    'SPM_design_column_names.tsv'), ...
    'FileType', 'text', 'Delimiter', '\t');
end
