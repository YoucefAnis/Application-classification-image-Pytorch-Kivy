from kivy.app import App
from kivy.uix.widget import Widget
from kivy.graphics import Line, Color
from kivy.uix.button import Button
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.popup import Popup
from kivy.uix.filechooser import FileChooserListView
from kivy.uix.textinput import TextInput


class SketchWidget(Widget):
    def on_touch_down(self, touch):
        with self.canvas:
            Color(1, 1, 1, 1)  # set canvas color to white
            touch.ud["line"] = Line(points=(touch.x, touch.y), width=2)
            Color(0, 0, 0, 0)  # set line color to black

    def on_touch_move(self, touch):
        touch.ud["line"].points += [touch.x, touch.y]

    def save_sketch(self, filename):
        self.export_to_png(filename)
        

class MyApp(App):
    def build(self):
        box = BoxLayout(orientation='vertical')
        self.sketch = SketchWidget()  # create named reference to SketchWidget instance
        button_layout = BoxLayout(orientation='horizontal', size_hint=(1, 0.1))
        button = Button(text='Save sketch', size_hint=(0.5, 1))
        button.bind(on_press=self.show_save_dialog)
        button_layout.add_widget(button)
        box.add_widget(self.sketch)  # use named reference in add_widget
        box.add_widget(button_layout)
        return box

    def show_save_dialog(self, instance):
        save_dialog = Popup(title="Save Sketch As...", size_hint=(0.9, 0.9))
        file_chooser = FileChooserListView(path=".")
        file_chooser.filters = ["*.png"]
        file_chooser.path = "./"
        file_chooser.multiselect = False
        save_button = Button(text="Save", size_hint=(1, 0.2))
        save_button.bind(on_press=lambda x: self.save_sketch(file_chooser.path + '/' + filename_input.text + '.png'))
        cancel_button = Button(text="Cancel", size_hint=(1, 0.2))
        cancel_button.bind(on_press=save_dialog.dismiss)
        filename_input = TextInput(text='', size_hint=(1, 0.2))
        save_dialog.content = BoxLayout(orientation="vertical")
        save_dialog.content.add_widget(filename_input)
        save_dialog.content.add_widget(file_chooser)
        button_layout = BoxLayout(orientation="horizontal", size_hint=(1, 0.2))
        button_layout.add_widget(save_button)
        button_layout.add_widget(cancel_button)
        save_dialog.content.add_widget(button_layout)
        save_dialog.open()
        
    def save_sketch(self, filename):
        self.sketch.save_sketch(filename)  # use named refere


if __name__ == '__main__':
    MyApp().run()
