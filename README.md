20240525 14:28
On Mac Apple Silicon:
```
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
```

The initial edge.py GPT4 write for me is likely for general pictures, which it use gray and blur into the image, but for screen shot I think that's not necessary. 
The next version code apply Sobel edge detection on each color channel, but this method also defines the kernel size. 
I just then use the differentiation by minus value of the pixel next to it, and it works well. But this method has not been accelerated, it do the pixel value one by one which is very slow.

The edge.py works well already.

I create a new file connected_components which I want to do the area segmentation first. 
My intial idea is to find the connected_components by the pixel value, and rank different value by the area size /pixel number. But the first version was extremely slow. 


20240525 17:15
TODO solve the error of connnected_top10_reconstruct.py
```
Processing color 4989/4990...
Processing color 4990/4990...
Sorting components by size...
Reconstructing image with top 10 components...
Traceback (most recent call last):
  File "/Users/chenyiyun/Desktop/YuantsyDesktopEdgeSegmentation/connected_top10_reconstruct.py", line 81, in <module>
    main()
  File "/Users/chenyiyun/Desktop/YuantsyDesktopEdgeSegmentation/connected_top10_reconstruct.py", line 75, in main
    reconstructed_image = reconstruct_image(image, components, unique_colors, rank_dict)
                          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/chenyiyun/Desktop/YuantsyDesktopEdgeSegmentation/connected_top10_reconstruct.py", line 60, in reconstruct_image
    for size, color_index, component_id in rank_dict.keys():
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
TypeError: cannot unpack non-iterable int object
```

