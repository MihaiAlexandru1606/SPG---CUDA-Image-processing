import cupy as cp

gray_scale = cp.RawKernel(r'''
                                extern "C"
                                __global__ void grayscale(int3 * mat, int width, int height){
                                    int row = blockIdx.y * blockDim.y + threadIdx.y;
                                    int col = blockIdx.x * blockDim.x + threadIdx.x;

                                    if (col < width && row < height) {
                                        int idx = row * width + col;
                                        int gray = 0.2125 * mat[idx].x + 0.7154 * mat[idx].y + 0.0721 * mat[idx].z;
                                        mat[idx].x = gray;
                                        mat[idx].y = gray;
                                        mat[idx].z = gray;
                                    }
                                }
                                ''', 'grayscale')

red = cp.RawKernel(r'''
                                extern "C"
                                __global__ void red(int3 * mat, int width, int height){
                                    int row = blockIdx.y * blockDim.y + threadIdx.y;
                                    int col = blockIdx.x * blockDim.x + threadIdx.x;

                                    if (col < width && row < height) {
                                        int idx = row * width + col;
                                        mat[idx].y = 0;
                                        mat[idx].z = 0;
                                    }
                                }
                                ''', 'red')

blue = cp.RawKernel(r'''
                                extern "C"
                                __global__ void blue(int3 * mat, int width, int height){
                                    int row = blockIdx.y * blockDim.y + threadIdx.y;
                                    int col = blockIdx.x * blockDim.x + threadIdx.x;

                                    if (col < width && row < height) {
                                        int idx = row * width + col;
                                        mat[idx].x = 0;
                                        mat[idx].y = 0;
                                    }
                                }
                                ''', 'blue')

green = cp.RawKernel(r'''
                                extern "C"
                                __global__ void green(int3 * mat, int width, int height){
                                    int row = blockIdx.y * blockDim.y + threadIdx.y;
                                    int col = blockIdx.x * blockDim.x + threadIdx.x;

                                    if (col < width && row < height) {
                                        int idx = row * width + col;
                                        mat[idx].x = 0;
                                        mat[idx].z = 0;
                                    }
                                }
                                ''', 'green')

red_remove = cp.RawKernel(r'''
                                extern "C"
                                __global__ void red_remove(int3 * mat, int width, int height){
                                    int row = blockIdx.y * blockDim.y + threadIdx.y;
                                    int col = blockIdx.x * blockDim.x + threadIdx.x;

                                    if (col < width && row < height) {
                                        int idx = row * width + col;
                                        mat[idx].x = 0;
                                        
                                    }
                                }
                                ''', 'red_remove')

blue_remove = cp.RawKernel(r'''
                                extern "C"
                                __global__ void blue_remove(int3 * mat, int width, int height){
                                    int row = blockIdx.y * blockDim.y + threadIdx.y;
                                    int col = blockIdx.x * blockDim.x + threadIdx.x;

                                    if (col < width && row < height) {
                                        int idx = row * width + col;
                                        mat[idx].z = 0;
                                    }
                                }
                                ''', 'blue_remove')

green_remove = cp.RawKernel(r'''
                                extern "C"
                                __global__ void green_remove(int3 * mat, int width, int height){
                                    int row = blockIdx.y * blockDim.y + threadIdx.y;
                                    int col = blockIdx.x * blockDim.x + threadIdx.x;

                                    if (col < width && row < height) {
                                        int idx = row * width + col;
                                        mat[idx].y = 0;
                                    }
                                }
                                ''', 'green_remove')

inverse = cp.RawKernel(r'''
                            extern "C"
                            __global__ void inverse(int3 * mat, int width, int height){
                                int row = blockIdx.y * blockDim.y + threadIdx.y;
                                int col = blockIdx.x * blockDim.x + threadIdx.x;

                                if (col < width && row < height) {
                                    int idx = row * width + col;
                                    mat[idx].x = 255 - mat[idx].x;
                                    mat[idx].y = 255 - mat[idx].y;
                                    mat[idx].z = 255 - mat[idx].z;
                                }
                            }
                            ''', 'inverse')

gaussian = cp.RawKernel(r'''
                            extern "C"
                                __global__ void gaussian(int3 * input, int3* output, int width, int height){
                                    int row = blockIdx.y * blockDim.y + threadIdx.y;
                                    int col = blockIdx.x * blockDim.x + threadIdx.x;
                                    int gaussian_kernel[3][3] = {
                                                                {1, 2, 1},
                                                                {2, 4, 2},
                                                                {1, 2, 1}
                                                                };
                                    
                                    int sum = 0;
                                    int pix_r = 0;
                                    int pix_g = 0;
                                    int pix_b = 0;
                                    
                                    if (col < width && row < height) {
                            
                                        for(int i = -1; i <= 1; i++) {
                                            for(int j = -1; j <= 1;j++) {
                                                if(col + j >= 0 && col + j < width && row + i >= 0 && row + i < height) {
                                                    int idx = (row + i) * width + col + j;
                                                    pix_r += gaussian_kernel[i + 1][j + 1] * input[idx].x;    
                                                    pix_g += gaussian_kernel[i + 1][j + 1] * input[idx].y;    
                                                    pix_b += gaussian_kernel[i + 1][j + 1] * input[idx].z;
                                                    sum += gaussian_kernel[i + 1][j + 1];
                                                }
                                            } 
                                        }
                                        
                                        int idx = row * width + col;
                                        output[idx].x = pix_r / sum;
                                        output[idx].y = pix_g / sum;
                                        output[idx].z = pix_b / sum;   
                                    }
                                }
                        ''', 'gaussian')

smooth = cp.RawKernel(r'''
                            extern "C"
                                __global__ void smooth(int3 * input, int3* output, int width, int height){
                                    int row = blockIdx.y * blockDim.y + threadIdx.y;
                                    int col = blockIdx.x * blockDim.x + threadIdx.x;
                                    float smooth_kernel[3][3] = {
                                                                {1.0f / 9.0f, 1.0f / 9.0f, 1.0f / 9.0f},
                                                                {1.0f / 9.0f, 1.0f / 9.0f, 1.0f / 9.0f},
                                                                {1.0f / 9.0f, 1.0f / 9.0f, 1.0f / 9.0f}
                                                                };

                                    int pix_r = 0;
                                    int pix_g = 0;
                                    int pix_b = 0;

                                    if (col < width && row < height) {

                                        for(int i = -1; i <= 1; i++) {
                                            for(int j = -1; j <= 1;j++) {
                                                if(col + j >= 0 && col + j < width && row + i >= 0 && row + i < height) {
                                                    int idx = (row + i) * width + col + j;
                                                    pix_r += smooth_kernel[i + 1][j + 1] * input[idx].x;    
                                                    pix_g += smooth_kernel[i + 1][j + 1] * input[idx].y;    
                                                    pix_b += smooth_kernel[i + 1][j + 1] * input[idx].z;
                                                }
                                            } 
                                        }

                                        int idx = row * width + col;
                                        output[idx].x = pix_r;
                                        output[idx].y = pix_g;
                                        output[idx].z = pix_b;   
                                    }
                                }
                        ''', 'smooth')

sharpen = cp.RawKernel(r'''
                            extern "C"
                                __global__ void sharpen(int3 * input, int3* output, int width, int height){
                                    int row = blockIdx.y * blockDim.y + threadIdx.y;
                                    int col = blockIdx.x * blockDim.x + threadIdx.x;
                                    float sharpen_kernel[3][3] = {
                                                                {0,     - 2.0f / 3.0f,  0},
                                                                {- 2.0f / 3.0f, 11.0f / 3, -2.0f / 3},
                                                                {0, -2.0f / 3, 0}
                                                                };

                                    int pix_r = 0;
                                    int pix_g = 0;
                                    int pix_b = 0;

                                    if (col < width && row < height) {

                                        for(int i = -1; i <= 1; i++) {
                                            for(int j = -1; j <= 1;j++) {
                                                if(col + j >= 0 && col + j < width && row + i >= 0 && row + i < height) {
                                                    int idx = (row + i) * width + col + j;
                                                    pix_r += sharpen_kernel[i + 1][j + 1] * input[idx].x;    
                                                    pix_g += sharpen_kernel[i + 1][j + 1] * input[idx].y;    
                                                    pix_b += sharpen_kernel[i + 1][j + 1] * input[idx].z;
                                                }
                                            } 
                                        }

                                        int idx = row * width + col;
                                        output[idx].x = pix_r;
                                        output[idx].y = pix_g;
                                        output[idx].z = pix_b;   
                                    }
                                }
                        ''', 'sharpen')

mean = cp.RawKernel(r'''
                            extern "C"
                                __global__ void mean(int3 * input, int3* output, int width, int height){
                                    int row = blockIdx.y * blockDim.y + threadIdx.y;
                                    int col = blockIdx.x * blockDim.x + threadIdx.x;
                                    float mean_kernel[3][3] = {
                                                                {-1.0f, -1.0f, -1.0f},
                                                                {-1.0f, 9.0f, -1.0f},
                                                                {-1.0f, -1.0f, -1.0f}
                                                                };

                                    int pix_r = 0;
                                    int pix_g = 0;
                                    int pix_b = 0;

                                    if (col < width && row < height) {

                                        for(int i = -1; i <= 1; i++) {
                                            for(int j = -1; j <= 1;j++) {
                                                if(col + j >= 0 && col + j < width && row + i >= 0 && row + i < height) {
                                                    int idx = (row + i) * width + col + j;
                                                    pix_r += mean_kernel[i + 1][j + 1] * input[idx].x;    
                                                    pix_g += mean_kernel[i + 1][j + 1] * input[idx].y;    
                                                    pix_b += mean_kernel[i + 1][j + 1] * input[idx].z;
                                                }
                                            } 
                                        }

                                        int idx = row * width + col;
                                         if(pix_r < 0)
                                            pix_r = 0;
                                        
                                        if(pix_g < 0)
                                            pix_g = 0;
                                        
                                        if(pix_b < 0)
                                            pix_b = 0;
                                        
                                        output[idx].x = pix_r;
                                        output[idx].y = pix_g;
                                        output[idx].z = pix_b;   
                                    }
                                }
                        ''', 'mean')

emboss = cp.RawKernel(r'''
                            extern "C"
                                __global__ void emboss(int3 * input, int3* output, int width, int height){
                                    int row = blockIdx.y * blockDim.y + threadIdx.y;
                                    int col = blockIdx.x * blockDim.x + threadIdx.x;
                                    float emboss_kernel[3][3] = {
                                                                {0, 1, 0},
                                                                {0, 0, 0},
                                                                {0, -1, 0}
                                                                };

                                    int pix_r = 0;
                                    int pix_g = 0;
                                    int pix_b = 0;

                                    if (col < width && row < height) {

                                        for(int i = -1; i <= 1; i++) {
                                            for(int j = -1; j <= 1;j++) {
                                                if(col + j >= 0 && col + j < width && row + i >= 0 && row + i < height) {
                                                    int idx = (row + i) * width + col + j;
                                                    pix_r += emboss_kernel[i + 1][j + 1] * input[idx].x;    
                                                    pix_g += emboss_kernel[i + 1][j + 1] * input[idx].y;    
                                                    pix_b += emboss_kernel[i + 1][j + 1] * input[idx].z;
                                                }
                                            } 
                                        }
                                        
                                        if(pix_r < 0)
                                            pix_r = 0;
                                        
                                        if(pix_g < 0)
                                            pix_g = 0;
                                        
                                        if(pix_b < 0)
                                            pix_b = 0;        
                                        int idx = row * width + col;
                                        output[idx].x = pix_r;
                                        output[idx].y = pix_g;
                                        output[idx].z = pix_b;   
                                    }
                                }
                        ''', 'emboss')

vignette = cp.RawKernel(r'''
                        extern "C"
                                __global__ void vignette(int3 * mat, int width, int height){
                                    int row = blockIdx.y * blockDim.y + threadIdx.y;
                                    int col = blockIdx.x * blockDim.x + threadIdx.x;
                                    
                                    if (col < width && row < height) {
                                        int idx = row * width + col;
                                        
                                        const double max_image_radius = 1.0 * sqrtf((height * height + width * width) / 4);
                                        const double power = 0.7;

                                        double dist_from_center = sqrtf((height / 2 - row) * (height / 2 - row) + (width / 2 - col) * (width / 2 - col));                                         
                                        dist_from_center *= power;
                                        dist_from_center = cos(dist_from_center) * cos(dist_from_center) * cos(dist_from_center) * cos(dist_from_center); 
                                        
                                        mat[idx].x = mat[idx].x * dist_from_center;
                                        mat[idx].y = mat[idx].y * dist_from_center;
                                        mat[idx].z = mat[idx].z * dist_from_center;
                                    }
                                }
                        ''', 'vignette')

contrast = cp.RawKernel(r'''
                                extern "C"
                                __global__ void contrast(int3 * mat, int width, int height, int contrast){
                                    int row = blockIdx.y * blockDim.y + threadIdx.y;
                                    int col = blockIdx.x * blockDim.x + threadIdx.x;

                                    if (col < width && row < height) {
                                        int idx = row * width + col;
                                        float factor = (259 * (contrast + 255)) / (255 * (259 - contrast));
                                        mat[idx].x = factor * (mat[idx].x - 128) + 128;
                                        mat[idx].y = factor * (mat[idx].y - 128) + 128;
                                        mat[idx].z = factor * (mat[idx].z - 128) + 128;
                                        
                                        if (mat[idx].x < 0) {
                                            mat[idx].x = 0;
                                        }
                                        
                                        if (mat[idx].y < 0) {
                                            mat[idx].y = 0;
                                        }
                                        
                                        if (mat[idx].z < 0) {
                                            mat[idx].z = 0;
                                        }
                                        
                                        if (mat[idx].x > 255) {
                                            mat[idx].x = 255;
                                        }
                                        
                                        if (mat[idx].y > 255) {
                                            mat[idx].y = 255;
                                        }
                                        
                                        if (mat[idx].z > 255) {
                                            mat[idx].z = 255;
                                        }
                                    }
                                }
                                ''', 'contrast')

sobel = cp.RawKernel(r'''
                            extern "C"
                                __global__ void sobel(int3 * input, int3* output, int width, int height){
                                    int row = blockIdx.y * blockDim.y + threadIdx.y;
                                    int col = blockIdx.x * blockDim.x + threadIdx.x;
                                    //const int Gx[3][3] = {{-1, 0, 1}, {-1, 0, 1}, {-1, 0, 1}};
                                    //const int Gy[3][3] = {{-1, -1, -1}, {0, 0, 0}, {1, 1, 1}};
                                    
                                    const int Gx[3][3] = {{-1, 0, 1}, {-2, 0, 2}, {-1, 0, 1}};
                                    const int Gy[3][3] = {{-1, -2, -1}, {0, 0, 0}, {1, 2, 1}};
                                    
                                    if (col < width && row < height) {
                                        float sum_x = 0;
                                        float sum_y = 0;
                                        
                                        for(int i = -1; i <= 1; i++) {
                                            for(int j = -1; j <= 1;j++) {
                                                if(col + j >= 0 && col + j < width && row + i >= 0 && row + i < height) {
                                                    int idx = (row + i) * width + col + j;
                                                    int gray = 0.2125 * input[idx].x + 0.7154 * input[idx].y + 0.0721 * input[idx].z;
                                                    
                                                    sum_x += Gx[i + 1][j + 1] * gray;    
                                                    sum_y += Gy[i + 1][j + 1] * gray;
                                                }
                                            } 
                                        }
                                        
                                        int idx = row * width + col;
                                        float sum = sqrt(sum_x * sum_x + sum_y * sum_y);
                                        if (sum > 255){
                                            sum = 255;
                                        }
                                        
                                        output[idx].x = sum;
                                        output[idx].y = sum;
                                        output[idx].z = sum;   
                                    }
                                }
                    
                    ''', 'sobel')

prewitt = cp.RawKernel(r'''
                            extern "C"
                                __global__ void prewitt(int3 * input, int3* output, int width, int height){
                                    int row = blockIdx.y * blockDim.y + threadIdx.y;
                                    int col = blockIdx.x * blockDim.x + threadIdx.x;
                                    const int Gx[3][3] = {{1, 0, -1}, {1, 0, -1}, {1, 0, -1}};
                                    const int Gy[3][3] = {{1, 1, 1}, {0, 0, 0}, {-1, -1, -1}};

                                    if (col < width && row < height) {
                                        float sum_x = 0;
                                        float sum_y = 0;

                                        for(int i = -1; i <= 1; i++) {
                                            for(int j = -1; j <= 1;j++) {
                                                if(col + j >= 0 && col + j < width && row + i >= 0 && row + i < height) {
                                                    int idx = (row + i) * width + col + j;
                                                    int gray = 0.2125 * input[idx].x + 0.7154 * input[idx].y + 0.0721 * input[idx].z;

                                                    sum_x += Gx[i + 1][j + 1] * gray;    
                                                    sum_y += Gy[i + 1][j + 1] * gray;
                                                }
                                            } 
                                        }

                                        int idx = row * width + col;
                                        float sum = abs(sum_x) + abs(sum_y);
                                        if (sum > 255){
                                            sum = 255;
                                        }

                                        output[idx].x = sum;
                                        output[idx].y = sum;
                                        output[idx].z = sum;   
                                    }
                                }

                    ''', 'prewitt')
