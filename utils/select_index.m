function [train_index, test_index] = select_index(dm_index, sti)

    switch(sti)
        case 1 % source:1,2 target:3,4
            train_index = [dm_index{1};dm_index{2}];
            test_index = [dm_index{3};dm_index{4}];
        case 2 % source:1,3 target:2,4
            train_index = [dm_index{1};dm_index{3}];
            test_index = [dm_index{2};dm_index{4}];
        case 3 % source:1,4 target:2,3
            train_index = [dm_index{1};dm_index{4}];
            test_index = [dm_index{2};dm_index{3}];
        case 4 % source:2,3 target:1,4
            train_index = [dm_index{2};dm_index{3}];
            test_index = [dm_index{1};dm_index{4}];
        case 5 % source:2,4 target:1,3
            train_index = [dm_index{2};dm_index{4}];
            test_index = [dm_index{1};dm_index{3}];
        case 6 % source:3,4 target:1,2
            train_index = [dm_index{3};dm_index{4}];
            test_index = [dm_index{1};dm_index{2}];
        case 7 % source:1,2,3 target:4
            train_index = [dm_index{1};dm_index{2};dm_index{3}];
            test_index = [dm_index{4}];
        case 8 % source:1,2,4 target:3
            train_index = [dm_index{1};dm_index{2};dm_index{4}];
            test_index = [dm_index{3}];
        case 9 % source:1,3,4 target:2
            train_index = [dm_index{1};dm_index{3};dm_index{4}];
            test_index = [dm_index{2}];
        case 10 % source:2,3,4 target:1
            train_index = [dm_index{2};dm_index{3};dm_index{4}];
            test_index = [dm_index{1}];
    end

end

