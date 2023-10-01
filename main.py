#---------------------------------------------------------------------------------------------------------
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.filechooser import FileChooserIconView
from kivy.uix.image import Image
from kivy.uix.label import Label
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
import torch
from torchsummary import summary
from model_fer_pytorch_test import Model
from model_mnist_pytorch_resnet_test import LeNet, im_convert, transform2, model3
from mnist_fashion_pytorch import Neural_Network, view_classify
import torchvision
import torchvision.transforms as transforms
from PIL import Image as Im
import PIL.ImageOps

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
device

class ClassificationApp(App):
    def __init__(self, **kwargs):
        super(ClassificationApp, self).__init__(**kwargs)
        self.model_file = None
        self.image_file = None
        self.result_label = Label(text='Sélectionnez un modèle et une image')
        self.image_widget = Image(source='', size_hint=(1, 1))
       
        
    def build(self):
        layout = BoxLayout(orientation='vertical')

        # Widget pour sélectionner un modèle au format .h5
        model_chooser = FileChooserIconView(path='.', filters=['*.h5', '*.pt'])
        model_chooser.bind(selection=self.on_model_selection)
        layout.add_widget(model_chooser)

        # Widget pour sélectionner une image
        image_chooser = FileChooserIconView(path='.', filters=['*.png', '*.jpg', '*.jpeg'])
        image_chooser.bind(selection=self.on_image_selection)
        layout.add_widget(image_chooser)
        
        # Widget pour afficher l'image sélectionnée
        layout.add_widget(self.image_widget)

        # Bouton pour lancer la classification
        classify_button = Button(text='Classify', on_press=self.classify, size_hint=(0.2,0.2), pos_hint={'center_x': 0.5, 'center_y': 0.5})
        layout.add_widget(classify_button)

        # Zone d'affichage du résultat
        layout.add_widget(self.result_label)

        return layout

    # Pour sélectionner et charger les poids d'un modèle
    def on_model_selection(self, instance, selection):
        if selection:
            self.model_file = selection[0]
            myModel = os.path.basename(self.model_file)
            if (myModel == "fer2013_pytorch.pt"):
                global model, height, width, channel, model_path
                model_path = self.model_file
                model = Model(1,7).to(device) 
                model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
                model.eval()
                print(model)
                print(selection[0])
                summary(model, (1, 48, 48))
            
            elif (myModel == 'mnist_pytorch.pt'):
                #global model, opt, height, width, channel, model_path
                global model2
                model_path = self.model_file
                model2= LeNet().to(device)
                print(model2)
                model2.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
                model2.eval()
                print(model2)
                print(selection[0])
                summary(model2, (1, 28, 28))

            elif (myModel == 'mnist_fashion_pytorch.pt'):
                #global model, opt, height, width, channel, model_path
                global model3
                model_path = self.model_file
                model3= Neural_Network().to(device)
                model3.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
                model3.eval()
                print(model3)
                print(selection[0])
                summary(model3, (1, 784))

#---------------------------------------------------------------------------------------------------------------
    def on_image_selection(self, instance, selection):
        global p
        if selection:
            self.image_file = selection[0]
            print("selection d'image fer et l'afficher dans l'interface", self.image_file)
            self.image_widget.source = self.image_file
            print(selection[0])
            p = selection[0]
        

    def classify(self, instance):
        if not self.model_file:
            self.result_label.text = 'Sélectionnez un modèle'
            return
        if not self.image_file:
            self.result_label.text = 'Sélectionnez une image'
            return
       
        if (os.path.basename(self.model_file) == "fer2013_pytorch.pt"):
            categories = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
            print(categories)
            y_pos = np.arange(len(categories))
        
            # Faire la prediction
            img_path = p
            img_path = os.path.join("./Test-FER2013", os.path.basename(img_path))
            image = cv2.imread(img_path,cv2.IMREAD_GRAYSCALE)
            print("shape de image avant transformation:",image.shape)
            transform = torchvision.transforms.Compose([
                        torchvision.transforms.ToPILImage(),
                        torchvision.transforms.Resize(48),
                        torchvision.transforms.Grayscale(num_output_channels=1),
                        torchvision.transforms.ToTensor(),
                        torchvision.transforms.Normalize((0.5), (0.5))])
            
            image = transform(image)
            print('shape de image apres transformation', image.shape)
            with torch.no_grad():
                output = model(image.unsqueeze(0))
                output_plot = output.tolist()
                print('output_plot',output_plot)
                print('output', output)
                _, predicted = torch.max(output.data, 1)
                print('predicted', predicted)
    
            print('Prediction:', categories[predicted.item()])
            print( "L'expression de cette image a une valeur de :" ,predicted.item())
            print( "L'expression de cette image est :" ,categories[predicted.item()])
    
            #Affichage des graphs
            plt.bar(y_pos, output_plot[0], align='center', alpha=0.5)
            plt.xticks(y_pos, categories)
            plt.ylabel('Percentage')
            plt.title('Facial Expression Prediction')
            plt.show()
            # Affichage du résultat
            self.result_label.text = f"L'expression de cette image est: {categories[predicted.item()]}"
        elif (os.path.basename(self.model_file) == "mnist_pytorch.pt"):
            categories2 = [0,1,2,3,4,5,6,7,8,9]
            y_pos2 = np.arange(len(categories2))
        
            # Faire la prediction
            img_path = p
            image2 = Im.open(img_path)
            #image2 = PIL.ImageOps.invert(image2)  # we use Image operations from PIL to invert(i.e. make white black and vice versa)
            image2 = PIL.ImageOps.invert(image2.convert('RGB'))
            image2 = image2.convert('1') # we convert from RGB to Gray
            image2 = transform2(image2) # Apply the transform funct we defined earlier to make our downloaded img same as what we trained on
            print('image apres transformation', image2)
            print('shape de image apres transformation', image2.shape)
            print('size de image apres transformation', image2.size)
            images2 = image2.to(device)  # As our model is in the device
            img2 = images2[0].unsqueeze(0).unsqueeze(0)
            print(('img2', img2.shape))

            with torch.no_grad():
                output2 = model2(img2)
                _, predicted2 = torch.max(output2, 1)
                print(predicted2.item())
            
            print('Prediction:', categories2[predicted2.item()])
            print( "Le chiffre dans cette image est un:" ,predicted2.item())
            print( "Le chiffre dans cette image est un:" ,categories2[predicted2.item()])
            #Affichage des graphs
            plt.bar(y_pos2, output2[0], align='center', alpha=0.5)
            plt.xticks(y_pos2, categories2)
            plt.ylabel('Percentage')
            plt.title('Mnist prediction')
            plt.show()
            # Affichage du résultat
            self.result_label.text = f"Le chiffre dans cette image est un : {categories2[predicted2.item()]}"
        elif (os.path.basename(self.model_file) == "mnist_fashion_pytorch.pt"):
            categories3 = ['T-shirt/top',
                            'Trouser',
                            'Pullover',
                            'Dress',
                            'Coat',
                            'Sandal',
                            'Shirt',
                            'Sneaker',
                            'Bag',
                            'Ankle Boot']
            print(categories3)
            y_pos = np.arange(len(categories3))
            img_path = p
            img_path = os.path.join("./Test-Fashion", os.path.basename(img_path))
            print('image path', img_path)
            fashion = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE).astype(np.float32)
            fashion =  cv2.resize(fashion, (28,28))
            print(type(fashion))
            fashion = torch.from_numpy(fashion)
            print(type(fashion))
            print(fashion.shape)
            fashion = fashion.reshape(1, fashion.shape[0], fashion.shape[1])
            print(fashion.shape)
            print(fashion.dtype)
            fashion = fashion.view(1, 784)
            with torch.no_grad():
                #logps = model3.forward(fashion)
                output3 = model3(fashion)
                _, predicted3 = torch.max(output3, 1)
                print(predicted3.item())
            
            print('Prediction:', categories3[predicted3.item()])
            print( "Cette image est un:" ,predicted3.item())
            print( "Cette image est un:" ,categories3[predicted3.item()])
            self.result_label.text = f"Cette image est: {categories3[predicted3.item()]}"

if __name__ == '__main__':
    ClassificationApp().run()
