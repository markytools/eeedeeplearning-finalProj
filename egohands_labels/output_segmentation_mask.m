%   Filter            Possible Values        Info
%   ----------        --------------------   ------------------------
%   'Location'        'OFFICE','COURTYARD',  Video background location
%                     'LIVINGROOM'
%
%   'Activity'        'CHESS','JENGA',       Activity in video
%                     'PUZZLE','CARDS'
%
%   'Viewer'          'B','S','T','H'        Identity of egocentric viewer
%
%   'Partner'         'B','S','T','H'        Identity of egocentric partner
locations = ["OFFICE", "COURTYARD", "LIVINGROOM"];
activities = ["CHESS", "JENGA", "PUZZLE", "CARDS"];
for loc = ["OFFICE", "COURTYARD", "LIVINGROOM"]
    for actvty = ["CHESS", "JENGA", "PUZZLE", "CARDS"]
        videos = getMetaBy('Location', loc, 'Activity', actvty);
        for a = 1:4
            vid_hand_masks = zeros(720, 1280, 100);
            for f = 1:100
                hand_mask = getSegmentationMask(videos(a), f, 'all');
                vid_hand_masks(:,:,f) = hand_mask;
            end
            savefile = strcat('OUTPUT_MASKS/',actvty,"_",loc,"_",int2str(a),".mat");
            save(savefile, 'vid_hand_masks');
        end
    end
end