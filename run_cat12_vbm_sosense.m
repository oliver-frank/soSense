function run_cat12_vbm_sosense()
% RUN_CAT12_VBM_SOSENSE
% Cross-sectional CAT12 VBM preprocessing for all SOSENSE T1w scans.
%
% Pipeline:
%   1) Recursively find sub-*_T1w.nii.gz / .nii below INPUT_ROOT
%   2) Copy/decompress them into OUTPUT_ROOT (source data remain untouched)
%   3) Run the installed CAT12.9 standard cross-sectional segmentation batch
%      with surface processing and unnecessary derivative outputs disabled
%   4) Retain modulated, normalized gray-matter maps: mwp1*.nii
%   5) Smooth mwp1 maps with 6 mm and 8 mm FWHM: s6mwp1* / s8mwp1*
%   6) Export TIV/GM/WM/CSF/WMH values and a processing-status table
%
% Run in MATLAB with:
%   run_cat12_vbm_sosense
%
% The script is designed for SPM12 + CAT12.9 and uses the CAT batch template
% shipped with the installed CAT12 version, minimizing version mismatch.

%% ------------------------------------------------------------------------
%  USER SETTINGS
%  ------------------------------------------------------------------------
input_root = ...
    '/zi/flstorage/group_csp/in_house_datasets/sosense/mri/derivatives/cat12.9';

spm_root = ...
    '/zi/software/Matlab/spm12_mit_cat12';

output_root = ...
    '/zi/flstorage/group_csp/analyses/paul.grube/projects/project_sosense/out/VBM';

% Re-copy/re-unzip the T1w files even when an output copy already exists.
overwrite_prepared_t1 = false;

% Re-run CAT12 even if previous outputs are present.
% CAT12's lazy mode is enabled when this is false.
rerun_cat12 = false;

% Re-create smoothed images if they already exist.
overwrite_smoothing = false;

%% ------------------------------------------------------------------------
%  INITIAL SETUP
%  ------------------------------------------------------------------------
if ~isfolder(input_root)
    error('Input directory does not exist: %s', input_root);
end
if ~isfolder(spm_root)
    error('SPM directory does not exist: %s', spm_root);
end
if ~exist(fullfile(spm_root, 'spm.m'), 'file')
    error(['spm.m was not found directly below spm_root. ', ...
           'Please check: %s'], spm_root);
end

make_dir(output_root);
log_dir = fullfile(output_root, 'logs');
make_dir(log_dir);

log_file = fullfile(log_dir, ...
    ['CAT12_VBM_' datestr(now, 'yyyymmdd_HHMMSS') '.log']);
diary(log_file);
diary_cleanup = onCleanup(@() diary('off')); %#ok<NASGU>

fprintf('\n============================================================\n');
fprintf('CAT12 VBM processing started: %s\n', datestr(now));
fprintf('Input root : %s\n', input_root);
fprintf('SPM root   : %s\n', spm_root);
fprintf('Output root: %s\n', output_root);
fprintf('============================================================\n\n');

% Find the CAT12 installation and its version-matched standard batch template.
cat_template = fullfile(spm_root, 'toolbox', 'cat12', 'standalone', ...
    'cat_standalone_segment.m');

if ~exist(cat_template, 'file')
    template_hits = dir(fullfile(spm_root, '**', ...
        'cat_standalone_segment.m'));
    if isempty(template_hits)
        error(['Could not find cat_standalone_segment.m below ', ...
               '%s.'], spm_root);
    elseif numel(template_hits) > 1
        fprintf('Several CAT templates found; using: %s\n', ...
            fullfile(template_hits(1).folder, template_hits(1).name));
    end
    cat_template = fullfile(template_hits(1).folder, template_hits(1).name);
end

cat12_dir = fileparts(fileparts(cat_template));
addpath(spm_root);
addpath(cat12_dir);

if ~exist('spm', 'file')
    error('SPM could not be added to the MATLAB path.');
end
if ~exist('cat12', 'file') && ~exist('spm_cat12', 'file')
    error('CAT12 could not be added to the MATLAB path.');
end

spm('defaults', 'FMRI');
spm_get_defaults('cmdline', true);
% CAT12 documentation recommends expert mode for batch processing.
if exist('cat12', 'file')
    cat12('expert');
end
spm_jobman('initcfg');

fprintf('SPM version: %s\n', spm('Ver'));
try
    fprintf('CAT version: %s\n', cat_version());
catch
    fprintf('CAT directory: %s\n', cat12_dir);
end

%% ------------------------------------------------------------------------
%  DISCOVER ALL T1w FILES
%  ------------------------------------------------------------------------
gz_hits  = dir(fullfile(input_root, '**', 'sub-*_T1w.nii.gz'));
nii_hits = dir(fullfile(input_root, '**', 'sub-*_T1w.nii'));
all_hits = [gz_hits; nii_hits];

if isempty(all_hits)
    error('No sub-*_T1w.nii.gz or sub-*_T1w.nii files found below %s.', ...
        input_root);
end

source_files = fullfile({all_hits.folder}', {all_hits.name}');
source_names = {all_hits.name}';
subject_ids = cell(size(source_names));

for i = 1:numel(source_names)
    token = regexp(source_names{i}, '^(sub-[^_]+).*_T1w\.nii(?:\.gz)?$', ...
        'tokens', 'once');
    if isempty(token)
        error('Could not derive subject ID from: %s', source_names{i});
    end
    subject_ids{i} = token{1};
end

% Never silently select between several T1w scans for the same subject.
[unique_subjects, ~, subject_index] = unique(subject_ids, 'stable');
subject_counts = accumarray(subject_index, 1);
duplicate_subjects = unique_subjects(subject_counts > 1);

if ~isempty(duplicate_subjects)
    fprintf('\nMultiple T1w files were found for these subjects:\n');
    for i = 1:numel(duplicate_subjects)
        fprintf('  %s\n', duplicate_subjects{i});
        rows = find(strcmp(subject_ids, duplicate_subjects{i}));
        for j = 1:numel(rows)
            fprintf('    %s\n', source_files{rows(j)});
        end
    end
    error(['Duplicate subject inputs detected. Resolve them before running ', ...
           'the VBM pipeline.']);
end

% Sort subjects for reproducible ordering.
[subject_ids, order] = sort(subject_ids);
source_files = source_files(order);
source_names = source_names(order);

fprintf('Found %d unique T1w scans.\n', numel(source_files));

%% ------------------------------------------------------------------------
%  COPY / DECOMPRESS INPUTS INTO OUTPUT_ROOT
%  ------------------------------------------------------------------------
% CAT's standard non-BIDS output then appears in OUTPUT_ROOT/mri, /report, etc.
prepared_files = cell(size(source_files));

for i = 1:numel(source_files)
    src = source_files{i};
    src_name = source_names{i};

    if endsWith(src_name, '.nii.gz')
        destination_name = erase(src_name, '.gz');
    else
        destination_name = src_name;
    end

    destination_file = fullfile(output_root, destination_name);
    prepared_files{i} = destination_file;

    if exist(destination_file, 'file') && ~overwrite_prepared_t1
        fprintf('[%3d/%3d] Prepared T1 exists, keeping: %s\n', ...
            i, numel(source_files), destination_name);
        continue;
    end

    fprintf('[%3d/%3d] Preparing: %s\n', ...
        i, numel(source_files), src);

    if endsWith(src_name, '.nii.gz')
        if exist(destination_file, 'file')
            delete(destination_file);
        end
        output_files = gunzip(src, output_root);
        if isempty(output_files) || ~exist(destination_file, 'file')
            error('gunzip failed for: %s', src);
        end
    else
        [ok, message] = copyfile(src, destination_file, 'f');
        if ~ok
            error('Could not copy %s: %s', src, message);
        end
    end
end

%% ------------------------------------------------------------------------
%  BUILD AND RUN CAT12 SEGMENTATION
%  ------------------------------------------------------------------------
clear matlabbatch

run(cat_template);

% Keep only the CAT segmentation module.
matlabbatch = matlabbatch(1);

cat_job = matlabbatch{1}.spm.tools.cat.estwrite;
cat_job.data = add_volume_index(prepared_files);

% Cross-sectional VBM output: modulated, spatially normalized GM.
cat_job.output.surface = 0;
cat_job.output.GM.native = 0;
cat_job.output.GM.warped = 0;
cat_job.output.GM.mod = 1;
cat_job.output.GM.dartel = 0;

% WM and CSF are still estimated internally and written to the XML report,
% but image derivatives are not required for standard GM VBM.
cat_job.output.WM.native = 0;
cat_job.output.WM.warped = 0;
cat_job.output.WM.mod = 0;
cat_job.output.WM.dartel = 0;

cat_job.output.CSF.native = 0;
cat_job.output.CSF.warped = 0;
cat_job.output.CSF.mod = 0;
cat_job.output.CSF.dartel = 0;

cat_job.output.WMH.native = 0;
cat_job.output.WMH.warped = 0;
cat_job.output.WMH.mod = 0;
cat_job.output.WMH.dartel = 0;

cat_job.output.SL.native = 0;
cat_job.output.SL.warped = 0;
cat_job.output.SL.mod = 0;
cat_job.output.SL.dartel = 0;

cat_job.output.TPMC.native = 0;
cat_job.output.TPMC.warped = 0;
cat_job.output.TPMC.mod = 0;
cat_job.output.TPMC.dartel = 0;

% Disable atlas/ROI maps and corrected-anatomical derivatives to save space.
atlas_names = fieldnames(cat_job.output.ROImenu.atlases);
for i = 1:numel(atlas_names)
    atlas_name = atlas_names{i};
    if strcmp(atlas_name, 'ownatlas')
        cat_job.output.ROImenu.atlases.(atlas_name) = {''};
    else
        cat_job.output.ROImenu.atlases.(atlas_name) = 0;
    end
end

cat_job.output.atlas.native = 0;
cat_job.output.label.native = 0;
cat_job.output.label.warped = 0;
cat_job.output.label.dartel = 0;
cat_job.output.bias.native = 0;
cat_job.output.bias.warped = 0;
cat_job.output.bias.dartel = 0;
cat_job.output.las.native = 0;
cat_job.output.las.warped = 0;
cat_job.output.las.dartel = 0;
cat_job.output.jacobianwarped = 0;
cat_job.output.warps = [0 0];
if isfield(cat_job.output, 'rmat')
    cat_job.output.rmat = 1;
end

% Continue past individual failures and generate volume QC PDFs/XML reports.
cat_job.extopts.admin.ignoreErrors = 1;
cat_job.extopts.admin.print = 2;
cat_job.extopts.admin.verb = 2;
cat_job.extopts.admin.lazy = double(~rerun_cat12);

matlabbatch{1}.spm.tools.cat.estwrite = cat_job;

batch_file = fullfile(log_dir, 'CAT12_segmentation_batch.mat');
save(batch_file, 'matlabbatch', 'source_files', 'prepared_files', ...
    'subject_ids');

fprintf('\nRunning CAT12 segmentation for %d subjects...\n', ...
    numel(prepared_files));
spm_jobman('run', matlabbatch);

%% ------------------------------------------------------------------------
%  LOCATE mwp1 OUTPUTS AND SMOOTH AT 6 mm / 8 mm
%  ------------------------------------------------------------------------
cat_output_dir = output_root;
mri_dir = fullfile(output_root, 'mri');
report_dir = fullfile(output_root, 'report');

if ~isfolder(mri_dir)
    error(['Expected CAT mri directory was not created: %s\n', ...
           'Check the log and the CAT output-folder setting.'], mri_dir);
end

mwp1_files = cell(size(subject_ids));
cat_success = false(size(subject_ids));

for i = 1:numel(subject_ids)
    [~, base_name] = fileparts(prepared_files{i});
    expected_file = fullfile(mri_dir, ['mwp1' base_name '.nii']);
    mwp1_files{i} = expected_file;
    cat_success(i) = exist(expected_file, 'file') == 2;
end

successful_mwp1 = mwp1_files(cat_success);
missing_subjects = subject_ids(~cat_success);

fprintf('\nCAT12 produced mwp1 images for %d/%d subjects.\n', ...
    numel(successful_mwp1), numel(subject_ids));

if ~isempty(missing_subjects)
    fprintf('Missing/failed subjects:\n');
    fprintf('  %s\n', missing_subjects{:});
end

if isempty(successful_mwp1)
    error('No mwp1 images were created; smoothing cannot proceed.');
end

smooth_images(successful_mwp1, 6, 's6', overwrite_smoothing);
smooth_images(successful_mwp1, 8, 's8', overwrite_smoothing);

%% ------------------------------------------------------------------------
%  EXPORT TIV AND PROCESSING STATUS
%  ------------------------------------------------------------------------
if isfolder(report_dir)
    export_cat_volumes(report_dir, ...
        fullfile(output_root, 'CAT12_TIV_GM_WM_CSF_WMH.tsv'));
else
    warning('CAT report directory not found; no TIV table exported: %s', ...
        report_dir);
end

s6_exists = false(size(subject_ids));
s8_exists = false(size(subject_ids));
s6_files = cell(size(subject_ids));
s8_files = cell(size(subject_ids));

for i = 1:numel(subject_ids)
    [mwp_dir, mwp_name, mwp_ext] = fileparts(mwp1_files{i});
    s6_files{i} = fullfile(mwp_dir, ['s6' mwp_name mwp_ext]);
    s8_files{i} = fullfile(mwp_dir, ['s8' mwp_name mwp_ext]);
    s6_exists(i) = exist(s6_files{i}, 'file') == 2;
    s8_exists(i) = exist(s8_files{i}, 'file') == 2;
end

status_table = table(subject_ids, source_files, prepared_files, ...
    mwp1_files, cat_success, s6_files, s6_exists, s8_files, s8_exists, ...
    'VariableNames', {'subject_id', 'source_T1w', 'prepared_T1w', ...
    'mwp1_file', 'CAT12_success', 's6mwp1_file', 's6_success', ...
    's8mwp1_file', 's8_success'});

status_file = fullfile(output_root, 'CAT12_processing_status.tsv');
writetable(status_table, status_file, 'FileType', 'text', ...
    'Delimiter', '\t');

fprintf('\n============================================================\n');
fprintf('CAT12 VBM processing finished: %s\n', datestr(now));
fprintf('CAT root    : %s\n', cat_output_dir);
fprintf('GM VBM      : %s/mwp1*.nii\n', mri_dir);
fprintf('6 mm images : %s/s6mwp1*.nii\n', mri_dir);
fprintf('8 mm images : %s/s8mwp1*.nii\n', mri_dir);
fprintf('Status table: %s\n', status_file);
fprintf('Log file    : %s\n', log_file);
fprintf('============================================================\n');
end

%% ========================================================================
%  LOCAL FUNCTIONS
%  ========================================================================
function make_dir(directory_name)
if ~isfolder(directory_name)
    [ok, message] = mkdir(directory_name);
    if ~ok
        error('Could not create directory %s: %s', directory_name, message);
    end
end
end

function indexed_files = add_volume_index(files)
indexed_files = cellfun(@(x) [x ',1'], files, 'UniformOutput', false);
indexed_files = indexed_files(:);
end

function smooth_images(input_files, kernel_mm, prefix, overwrite_existing)
% Smooth only files whose requested output is missing, unless overwrite=true.
to_smooth = {};

for i = 1:numel(input_files)
    [folder_name, file_name, extension] = fileparts(input_files{i});
    output_file = fullfile(folder_name, [prefix file_name extension]);

    if overwrite_existing && exist(output_file, 'file')
        delete(output_file);
    end

    if ~exist(output_file, 'file')
        to_smooth{end + 1, 1} = input_files{i}; %#ok<AGROW>
    end
end

if isempty(to_smooth)
    fprintf('%s smoothing already complete; nothing to do.\n', prefix);
    return;
end

clear matlabbatch
matlabbatch{1}.spm.spatial.smooth.data = add_volume_index(to_smooth);
matlabbatch{1}.spm.spatial.smooth.fwhm = ...
    [kernel_mm kernel_mm kernel_mm];
matlabbatch{1}.spm.spatial.smooth.dtype = 0;
matlabbatch{1}.spm.spatial.smooth.im = 0;
matlabbatch{1}.spm.spatial.smooth.prefix = prefix;

fprintf('Smoothing %d images with %d mm FWHM (%s)...\n', ...
    numel(to_smooth), kernel_mm, prefix);
spm_jobman('run', matlabbatch);
end

function export_cat_volumes(report_dir, output_file)
% Export TIV and tissue volumes from CAT XML reports.
xml_hits = dir(fullfile(report_dir, 'cat_*.xml'));

if isempty(xml_hits)
    warning('No CAT XML reports found in %s.', report_dir);
    return;
end

xml_files = fullfile({xml_hits.folder}', {xml_hits.name}');
subject_id = cell(numel(xml_files), 1);
TIV = nan(numel(xml_files), 1);
GM = nan(numel(xml_files), 1);
WM = nan(numel(xml_files), 1);
CSF = nan(numel(xml_files), 1);
WMH = nan(numel(xml_files), 1);

for i = 1:numel(xml_files)
    [~, xml_name] = fileparts(xml_files{i});
    token = regexp(xml_name, '^cat_(sub-[^_]+)', 'tokens', 'once');
    if isempty(token)
        subject_id{i} = xml_name;
    else
        subject_id{i} = token{1};
    end

    try
        xml = cat_io_xml(xml_files{i});
        volumes = double(xml.subjectmeasures.vol_abs_CGW(:)');

        % CAT ordering used by cat_stat_TIV:
        % volumes(1)=CSF, volumes(2)=GM, volumes(3)=WM, volumes(4)=WMH.
        TIV(i) = sum(volumes, 'omitnan');
        if numel(volumes) >= 1, CSF(i) = volumes(1); end
        if numel(volumes) >= 2, GM(i)  = volumes(2); end
        if numel(volumes) >= 3, WM(i)  = volumes(3); end
        if numel(volumes) >= 4, WMH(i) = volumes(4); end
    catch ME
        warning('Could not read volumes from %s: %s', ...
            xml_files{i}, ME.message);
    end
end

volume_table = table(subject_id, TIV, GM, WM, CSF, WMH);
volume_table = sortrows(volume_table, 'subject_id');
writetable(volume_table, output_file, 'FileType', 'text', ...
    'Delimiter', '\t');
fprintf('TIV/tissue-volume table written: %s\n', output_file);
end
