jsonData = jsondecode(fileread('label_2.json'));
jsonDataFieldnames = fieldnames(jsonData);
for i = 1:length(jsonDataFieldnames)
    field_name = char(jsonDataFieldnames(i));
%     output_val = getfield(jsonData, field_name);
    game_loc_str = extractAfter(field_name, "LABELED_IMAGES_");
    game_loc_str = game_loc_str(1:end-15); %  ex: 'JENGA_LIVINGROOM_B_H'
    frame_name = field_name(end-13:end);
    frame_name = frame_name(1:10); % ex: 'frame_0637',  no '.jpg' string
    full_file_name = strcat('LABELED_IMAGES/',game_loc_str,'/',frame_name,'.jpg');
    
    hand = imread(strcat('./',full_file_name)); imageSegmenter(hand)
    n = 0.3;
    while ~exist('BW')
       pause(n)
    end
    output_file_name = strcat(game_loc_str,'_',frame_name,'.jpg');
    imwrite(BW, strcat('OCCLUDED_Y/',output_file_name)); clear BW;
end