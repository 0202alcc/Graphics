from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.label import Label
from kivy.uix.button import Button

class MainApp(App):
    def build(self):
        layout = BoxLayout(orientation='vertical', padding=10, spacing=10)
        label = Label(text='Welcome to Graphics App!')
        button = Button(text='Click me!')
        button.bind(on_press=self.on_button_press)
        layout.add_widget(label)
        layout.add_widget(button)
        return layout
    
    def on_button_press(self, instance):
        instance.text = 'Button pressed!'

if __name__ == '__main__':
    MainApp().run()