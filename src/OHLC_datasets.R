ohlc_dataset <- dataset(
  "OHLC",
  
  initialize = function(data_path, split = "train",
                        transform = NULL,
                        target_transform = NULL,
                        indexes = NULL) {
    self$transform <- transform
    self$target_transform <- target_transform
    
    # images and target variables
    if(split == "train") {
      self$images <- data.table::fread( fs::path(data_path, "train.csv") )
      if(!is.null(indexes)) self$images <- self$images[indexes, ]
      self$.path <- file.path(data_path, "train_imgs")
    } else if(split == "test") {
      self$images <- data.table::fread( fs::path(data_path, "test.csv") )
      if(!is.null(indexes)) self$images <- self$images[indexes, ]
      self$.path <- file.path(data_path, "test_imgs")
    }
  },
  
  
  .getitem = function(index) {
    
    force(index)
    
    sample <- self$images[index, ]
    id <- sample$id
    x <- torchvision::base_loader(file.path(self$.path, paste0(sample$id, ".png")))
    x <- torchvision::transform_to_tensor(x) %>% torchvision::transform_rgb_to_grayscale()
    
    if (!is.null(self$transform))
      x <- self$transform(x)
    
    y <- torch::torch_scalar_tensor(sample$y)
    if (!is.null(self$target_transform))
      y <- self$target_transform(y)
    
    return(list(x = x, y = y, id = id))
  },
  
  
  .length = function() {
    nrow(self$images)
  }
)